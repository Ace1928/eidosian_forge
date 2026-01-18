import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
class BotocoreCRTRequestSerializer(BaseCRTRequestSerializer):

    def __init__(self, session, client_kwargs=None):
        """Serialize CRT HTTP request using botocore logic
        It also takes into account configuration from both the session
        and any keyword arguments that could be passed to
        `Session.create_client()` when serializing the request.

        :type session: botocore.session.Session

        :type client_kwargs: Optional[Dict[str, str]])
        :param client_kwargs: The kwargs for the botocore
            s3 client initialization.
        """
        self._session = session
        if client_kwargs is None:
            client_kwargs = {}
        self._resolve_client_config(session, client_kwargs)
        self._client = session.create_client(**client_kwargs)
        self._client.meta.events.register('request-created.s3.*', self._capture_http_request)
        self._client.meta.events.register('after-call.s3.*', self._change_response_to_serialized_http_request)
        self._client.meta.events.register('before-send.s3.*', self._make_fake_http_response)

    def _resolve_client_config(self, session, client_kwargs):
        user_provided_config = None
        if session.get_default_client_config():
            user_provided_config = session.get_default_client_config()
        if 'config' in client_kwargs:
            user_provided_config = client_kwargs['config']
        client_config = Config(signature_version=UNSIGNED)
        if user_provided_config:
            client_config = user_provided_config.merge(client_config)
        client_kwargs['config'] = client_config
        client_kwargs['service_name'] = 's3'

    def _crt_request_from_aws_request(self, aws_request):
        url_parts = urlsplit(aws_request.url)
        crt_path = url_parts.path
        if url_parts.query:
            crt_path = f'{crt_path}?{url_parts.query}'
        headers_list = []
        for name, value in aws_request.headers.items():
            if isinstance(value, str):
                headers_list.append((name, value))
            else:
                headers_list.append((name, str(value, 'utf-8')))
        crt_headers = awscrt.http.HttpHeaders(headers_list)
        crt_request = awscrt.http.HttpRequest(method=aws_request.method, path=crt_path, headers=crt_headers, body_stream=aws_request.body)
        return crt_request

    def _convert_to_crt_http_request(self, botocore_http_request):
        crt_request = self._crt_request_from_aws_request(botocore_http_request)
        if crt_request.headers.get('host') is None:
            url_parts = urlsplit(botocore_http_request.url)
            crt_request.headers.set('host', url_parts.netloc)
        if crt_request.headers.get('Content-MD5') is not None:
            crt_request.headers.remove('Content-MD5')
        if crt_request.headers.get('Content-Length') is None:
            if botocore_http_request.body is None:
                crt_request.headers.add('Content-Length', '0')
        if crt_request.headers.get('Transfer-Encoding') is not None:
            crt_request.headers.remove('Transfer-Encoding')
        return crt_request

    def _capture_http_request(self, request, **kwargs):
        request.context['http_request'] = request

    def _change_response_to_serialized_http_request(self, context, parsed, **kwargs):
        request = context['http_request']
        parsed['HTTPRequest'] = request.prepare()

    def _make_fake_http_response(self, request, **kwargs):
        return botocore.awsrequest.AWSResponse(None, 200, {}, FakeRawResponse(b''))

    def _get_botocore_http_request(self, client_method, call_args):
        return getattr(self._client, client_method)(Bucket=call_args.bucket, Key=call_args.key, **call_args.extra_args)['HTTPRequest']

    def serialize_http_request(self, transfer_type, future):
        botocore_http_request = self._get_botocore_http_request(transfer_type, future.meta.call_args)
        crt_request = self._convert_to_crt_http_request(botocore_http_request)
        return crt_request

    def translate_crt_exception(self, exception):
        if isinstance(exception, awscrt.s3.S3ResponseError):
            return self._translate_crt_s3_response_error(exception)
        else:
            return None

    def _translate_crt_s3_response_error(self, s3_response_error):
        status_code = s3_response_error.status_code
        if status_code < 301:
            return None
        headers = {k: v for k, v in s3_response_error.headers}
        operation_name = s3_response_error.operation_name
        if operation_name is not None:
            service_model = self._client.meta.service_model
            shape = service_model.operation_model(operation_name).output_shape
        else:
            shape = None
        response_dict = {'headers': botocore.awsrequest.HeadersDict(headers), 'status_code': status_code, 'body': s3_response_error.body}
        parsed_response = self._client._response_parser.parse(response_dict, shape=shape)
        error_code = parsed_response.get('Error', {}).get('Code')
        error_class = self._client.exceptions.from_code(error_code)
        return error_class(parsed_response, operation_name=operation_name)