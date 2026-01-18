import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
class BaseApiService(object):
    """Base class for generated API services."""

    def __init__(self, client):
        self.__client = client
        self._method_configs = {}
        self._upload_configs = {}

    @property
    def _client(self):
        return self.__client

    @property
    def client(self):
        return self.__client

    def GetMethodConfig(self, method):
        """Returns service cached method config for given method."""
        method_config = self._method_configs.get(method)
        if method_config:
            return method_config
        func = getattr(self, method, None)
        if func is None:
            raise KeyError(method)
        method_config = getattr(func, 'method_config', None)
        if method_config is None:
            raise KeyError(method)
        self._method_configs[method] = config = method_config()
        return config

    @classmethod
    def GetMethodsList(cls):
        return [f.__name__ for f in six.itervalues(cls.__dict__) if getattr(f, 'method_config', None)]

    def GetUploadConfig(self, method):
        return self._upload_configs.get(method)

    def GetRequestType(self, method):
        method_config = self.GetMethodConfig(method)
        return getattr(self.client.MESSAGES_MODULE, method_config.request_type_name)

    def GetResponseType(self, method):
        method_config = self.GetMethodConfig(method)
        return getattr(self.client.MESSAGES_MODULE, method_config.response_type_name)

    def __CombineGlobalParams(self, global_params, default_params):
        """Combine the given params with the defaults."""
        util.Typecheck(global_params, (type(None), self.__client.params_type))
        result = self.__client.params_type()
        global_params = global_params or self.__client.params_type()
        for field in result.all_fields():
            value = global_params.get_assigned_value(field.name)
            if value is None:
                value = default_params.get_assigned_value(field.name)
            if value not in (None, [], ()):
                setattr(result, field.name, value)
        return result

    def __EncodePrettyPrint(self, query_info):
        if not query_info.pop('prettyPrint', True):
            query_info['prettyPrint'] = 0
        if not query_info.pop('pp', True):
            query_info['pp'] = 0
        return query_info

    def __FinalUrlValue(self, value, field):
        """Encode value for the URL, using field to skip encoding for bytes."""
        if isinstance(field, messages.BytesField) and value is not None:
            return base64.urlsafe_b64encode(value)
        elif isinstance(value, six.text_type):
            return value.encode('utf8')
        elif isinstance(value, six.binary_type):
            return value.decode('utf8')
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        return value

    def __ConstructQueryParams(self, query_params, request, global_params):
        """Construct a dictionary of query parameters for this request."""
        global_params = self.__CombineGlobalParams(global_params, self.__client.global_params)
        global_param_names = util.MapParamNames([x.name for x in self.__client.params_type.all_fields()], self.__client.params_type)
        global_params_type = type(global_params)
        query_info = dict(((param, self.__FinalUrlValue(getattr(global_params, param), getattr(global_params_type, param))) for param in global_param_names))
        query_param_names = util.MapParamNames(query_params, type(request))
        request_type = type(request)
        query_info.update(((param, self.__FinalUrlValue(getattr(request, param, None), getattr(request_type, param))) for param in query_param_names))
        query_info = dict(((k, v) for k, v in query_info.items() if v is not None))
        query_info = self.__EncodePrettyPrint(query_info)
        query_info = util.MapRequestParams(query_info, type(request))
        return query_info

    def __ConstructRelativePath(self, method_config, request, relative_path=None):
        """Determine the relative path for request."""
        python_param_names = util.MapParamNames(method_config.path_params, type(request))
        params = dict([(param, getattr(request, param, None)) for param in python_param_names])
        params = util.MapRequestParams(params, type(request))
        return util.ExpandRelativePath(method_config, params, relative_path=relative_path)

    def __FinalizeRequest(self, http_request, url_builder):
        """Make any final general adjustments to the request."""
        if http_request.http_method == 'GET' and len(http_request.url) > _MAX_URL_LENGTH:
            http_request.http_method = 'POST'
            http_request.headers['x-http-method-override'] = 'GET'
            http_request.headers['content-type'] = 'application/x-www-form-urlencoded'
            http_request.body = url_builder.query
            url_builder.query_params = {}
        http_request.url = url_builder.url

    def __ProcessHttpResponse(self, method_config, http_response, request):
        """Process the given http response."""
        if http_response.status_code not in (http_client.OK, http_client.CREATED, http_client.NO_CONTENT):
            raise exceptions.HttpError.FromResponse(http_response, method_config=method_config, request=request)
        if http_response.status_code == http_client.NO_CONTENT:
            http_response = http_wrapper.Response(info=http_response.info, content='{}', request_url=http_response.request_url)
        content = http_response.content
        if self._client.response_encoding and isinstance(content, bytes):
            content = content.decode(self._client.response_encoding)
        if self.__client.response_type_model == 'json':
            return content
        response_type = _LoadClass(method_config.response_type_name, self.__client.MESSAGES_MODULE)
        return self.__client.DeserializeMessage(response_type, content)

    def __SetBaseHeaders(self, http_request, client):
        """Fill in the basic headers on http_request."""
        user_agent = client.user_agent or 'apitools-client/1.0'
        http_request.headers['user-agent'] = user_agent
        http_request.headers['accept'] = 'application/json'
        http_request.headers['accept-encoding'] = 'gzip, deflate'

    def __SetBody(self, http_request, method_config, request, upload):
        """Fill in the body on http_request."""
        if not method_config.request_field:
            return
        request_type = _LoadClass(method_config.request_type_name, self.__client.MESSAGES_MODULE)
        if method_config.request_field == REQUEST_IS_BODY:
            body_value = request
            body_type = request_type
        else:
            body_value = getattr(request, method_config.request_field)
            body_field = request_type.field_by_name(method_config.request_field)
            util.Typecheck(body_field, messages.MessageField)
            body_type = body_field.type
        body_value = body_value or body_type()
        if upload and (not body_value):
            return
        util.Typecheck(body_value, body_type)
        http_request.headers['content-type'] = 'application/json'
        http_request.body = self.__client.SerializeMessage(body_value)

    def PrepareHttpRequest(self, method_config, request, global_params=None, upload=None, upload_config=None, download=None):
        """Prepares an HTTP request to be sent."""
        request_type = _LoadClass(method_config.request_type_name, self.__client.MESSAGES_MODULE)
        util.Typecheck(request, request_type)
        request = self.__client.ProcessRequest(method_config, request)
        http_request = http_wrapper.Request(http_method=method_config.http_method)
        self.__SetBaseHeaders(http_request, self.__client)
        self.__SetBody(http_request, method_config, request, upload)
        url_builder = _UrlBuilder(self.__client.url, relative_path=method_config.relative_path)
        url_builder.query_params = self.__ConstructQueryParams(method_config.query_params, request, global_params)
        if upload is not None:
            upload.ConfigureRequest(upload_config, http_request, url_builder)
        if download is not None:
            download.ConfigureRequest(http_request, url_builder)
        url_builder.relative_path = self.__ConstructRelativePath(method_config, request, relative_path=url_builder.relative_path)
        self.__FinalizeRequest(http_request, url_builder)
        return self.__client.ProcessHttpRequest(http_request)

    def _RunMethod(self, method_config, request, global_params=None, upload=None, upload_config=None, download=None):
        """Call this method with request."""
        if upload is not None and download is not None:
            raise exceptions.NotYetImplementedError('Cannot yet use both upload and download at once')
        http_request = self.PrepareHttpRequest(method_config, request, global_params, upload, upload_config, download)
        if download is not None:
            download.InitializeDownload(http_request, client=self.client)
            return
        http_response = None
        if upload is not None:
            http_response = upload.InitializeUpload(http_request, client=self.client)
        if http_response is None:
            http = self.__client.http
            if upload and upload.bytes_http:
                http = upload.bytes_http
            opts = {'retries': self.__client.num_retries, 'max_retry_wait': self.__client.max_retry_wait}
            if self.__client.check_response_func:
                opts['check_response_func'] = self.__client.check_response_func
            if self.__client.retry_func:
                opts['retry_func'] = self.__client.retry_func
            http_response = http_wrapper.MakeRequest(http, http_request, **opts)
        return self.ProcessHttpResponse(method_config, http_response, request)

    def ProcessHttpResponse(self, method_config, http_response, request=None):
        """Convert an HTTP response to the expected message type."""
        return self.__client.ProcessResponse(method_config, self.__ProcessHttpResponse(method_config, http_response, request))