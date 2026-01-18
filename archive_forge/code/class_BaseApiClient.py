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
class BaseApiClient(object):
    """Base class for client libraries."""
    MESSAGES_MODULE = None
    _API_KEY = ''
    _CLIENT_ID = ''
    _CLIENT_SECRET = ''
    _PACKAGE = ''
    _SCOPES = []
    _USER_AGENT = ''

    def __init__(self, url, credentials=None, get_credentials=True, http=None, model=None, log_request=False, log_response=False, num_retries=5, max_retry_wait=60, credentials_args=None, default_global_params=None, additional_http_headers=None, check_response_func=None, retry_func=None, response_encoding=None):
        _RequireClassAttrs(self, ('_package', '_scopes', 'messages_module'))
        if default_global_params is not None:
            util.Typecheck(default_global_params, self.params_type)
        self.__default_global_params = default_global_params
        self.log_request = log_request
        self.log_response = log_response
        self.__num_retries = 5
        self.__max_retry_wait = 60
        self.num_retries = num_retries
        self.max_retry_wait = max_retry_wait
        self._credentials = credentials
        get_credentials = get_credentials and (not _SkipGetCredentials())
        if get_credentials and (not credentials):
            credentials_args = credentials_args or {}
            self._SetCredentials(**credentials_args)
        self._url = NormalizeApiEndpoint(url)
        self._http = http or http_wrapper.GetHttp()
        if self._credentials is not None:
            self._http = self._credentials.authorize(self._http)
        self.__include_fields = None
        self.additional_http_headers = additional_http_headers or {}
        self.check_response_func = check_response_func
        self.retry_func = retry_func
        self.response_encoding = response_encoding
        self.overwrite_transfer_urls_with_client_base = False
        _ = model
        self.__response_type_model = 'proto'

    def _SetCredentials(self, **kwds):
        """Fetch credentials, and set them for this client.

        Note that we can't simply return credentials, since creating them
        may involve side-effecting self.

        Args:
          **kwds: Additional keyword arguments are passed on to GetCredentials.

        Returns:
          None. Sets self._credentials.
        """
        args = {'api_key': self._API_KEY, 'client': self, 'client_id': self._CLIENT_ID, 'client_secret': self._CLIENT_SECRET, 'package_name': self._PACKAGE, 'scopes': self._SCOPES, 'user_agent': self._USER_AGENT}
        args.update(kwds)
        from apitools.base.py import credentials_lib
        self._credentials = credentials_lib.GetCredentials(**args)

    @classmethod
    def ClientInfo(cls):
        return {'client_id': cls._CLIENT_ID, 'client_secret': cls._CLIENT_SECRET, 'scope': ' '.join(sorted(util.NormalizeScopes(cls._SCOPES))), 'user_agent': cls._USER_AGENT}

    @property
    def base_model_class(self):
        return None

    @property
    def http(self):
        return self._http

    @property
    def url(self):
        return self._url

    @classmethod
    def GetScopes(cls):
        return cls._SCOPES

    @property
    def params_type(self):
        return _LoadClass('StandardQueryParameters', self.MESSAGES_MODULE)

    @property
    def user_agent(self):
        return self._USER_AGENT

    @property
    def _default_global_params(self):
        if self.__default_global_params is None:
            self.__default_global_params = self.params_type()
        return self.__default_global_params

    def AddGlobalParam(self, name, value):
        params = self._default_global_params
        setattr(params, name, value)

    @property
    def global_params(self):
        return encoding.CopyProtoMessage(self._default_global_params)

    @contextlib.contextmanager
    def IncludeFields(self, include_fields):
        self.__include_fields = include_fields
        yield
        self.__include_fields = None

    @property
    def response_type_model(self):
        return self.__response_type_model

    @contextlib.contextmanager
    def JsonResponseModel(self):
        """In this context, return raw JSON instead of proto."""
        old_model = self.response_type_model
        self.__response_type_model = 'json'
        yield
        self.__response_type_model = old_model

    @property
    def num_retries(self):
        return self.__num_retries

    @num_retries.setter
    def num_retries(self, value):
        util.Typecheck(value, six.integer_types)
        if value < 0:
            raise exceptions.InvalidDataError('Cannot have negative value for num_retries')
        self.__num_retries = value

    @property
    def max_retry_wait(self):
        return self.__max_retry_wait

    @max_retry_wait.setter
    def max_retry_wait(self, value):
        util.Typecheck(value, six.integer_types)
        if value <= 0:
            raise exceptions.InvalidDataError('max_retry_wait must be a postiive integer')
        self.__max_retry_wait = value

    @contextlib.contextmanager
    def WithRetries(self, num_retries):
        old_num_retries = self.num_retries
        self.num_retries = num_retries
        yield
        self.num_retries = old_num_retries

    def ProcessRequest(self, method_config, request):
        """Hook for pre-processing of requests."""
        if self.log_request:
            logging.info('Calling method %s with %s: %s', method_config.method_id, method_config.request_type_name, request)
        return request

    def ProcessHttpRequest(self, http_request):
        """Hook for pre-processing of http requests."""
        http_request.headers.update(self.additional_http_headers)
        if self.log_request:
            logging.info('Making http %s to %s', http_request.http_method, http_request.url)
            logging.info('Headers: %s', pprint.pformat(http_request.headers))
            if http_request.body:
                logging.info('Body:\n%s', http_request.loggable_body or http_request.body)
            else:
                logging.info('Body: (none)')
        return http_request

    def ProcessResponse(self, method_config, response):
        if self.log_response:
            logging.info('Response of type %s: %s', method_config.response_type_name, response)
        return response

    def SerializeMessage(self, message):
        return encoding.MessageToJson(message, include_fields=self.__include_fields)

    def DeserializeMessage(self, response_type, data):
        """Deserialize the given data as method_config.response_type."""
        try:
            message = encoding.JsonToMessage(response_type, data)
        except (exceptions.InvalidDataFromServerError, messages.ValidationError, ValueError) as e:
            raise exceptions.InvalidDataFromServerError('Error decoding response "%s" as type %s: %s' % (data, response_type.__name__, e))
        return message

    def FinalizeTransferUrl(self, url):
        """Modify the url for a given transfer, based on auth and version."""
        url_builder = _UrlBuilder.FromUrl(url)
        if getattr(self.global_params, 'key', None):
            url_builder.query_params['key'] = self.global_params.key
        if self.overwrite_transfer_urls_with_client_base:
            client_url_builder = _UrlBuilder.FromUrl(self._url)
            url_builder.base_url = client_url_builder.base_url
        return url_builder.url