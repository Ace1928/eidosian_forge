from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
class AsyncAWSQueryConnection(AsyncConnection):
    """Async AWS Query Connection."""
    STATUS_CODE_OK = 200
    STATUS_CODE_REQUEST_TIMEOUT = 408
    STATUS_CODE_NETWORK_CONNECT_TIMEOUT_ERROR = 599
    STATUS_CODE_INTERNAL_ERROR = 500
    STATUS_CODE_BAD_GATEWAY = 502
    STATUS_CODE_SERVICE_UNAVAILABLE_ERROR = 503
    STATUS_CODE_GATEWAY_TIMEOUT = 504
    STATUS_CODES_SERVER_ERRORS = (STATUS_CODE_INTERNAL_ERROR, STATUS_CODE_BAD_GATEWAY, STATUS_CODE_SERVICE_UNAVAILABLE_ERROR)
    STATUS_CODES_TIMEOUT = (STATUS_CODE_REQUEST_TIMEOUT, STATUS_CODE_NETWORK_CONNECT_TIMEOUT_ERROR, STATUS_CODE_GATEWAY_TIMEOUT)

    def __init__(self, sqs_connection, http_client=None, http_client_params=None, **kwargs):
        if not http_client_params:
            http_client_params = {}
        super().__init__(sqs_connection, http_client, **http_client_params)

    def make_request(self, operation, params_, path, verb, callback=None):
        params = params_.copy()
        if operation:
            params['Action'] = operation
        signer = self.sqs_connection._request_signer
        signing_type = 'standard'
        param_payload = {'data': params}
        if verb.lower() == 'get':
            signing_type = 'presign-url'
            param_payload = {'params': params}
        request = AWSRequest(method=verb, url=path, **param_payload)
        signer.sign(operation, request, signing_type=signing_type)
        prepared_request = request.prepare()
        return self._mexe(prepared_request, callback=callback)

    def get_list(self, operation, params, markers, path='/', parent=None, verb='POST', callback=None):
        return self.make_request(operation, params, path, verb, callback=transform(self._on_list_ready, callback, parent or self, markers, operation))

    def get_object(self, operation, params, path='/', parent=None, verb='GET', callback=None):
        return self.make_request(operation, params, path, verb, callback=transform(self._on_obj_ready, callback, parent or self, operation))

    def get_status(self, operation, params, path='/', parent=None, verb='GET', callback=None):
        return self.make_request(operation, params, path, verb, callback=transform(self._on_status_ready, callback, parent or self, operation))

    def _on_list_ready(self, parent, markers, operation, response):
        service_model = self.sqs_connection.meta.service_model
        if response.status == self.STATUS_CODE_OK:
            _, parsed = get_response(service_model.operation_model(operation), response.response)
            return parsed
        elif response.status in self.STATUS_CODES_TIMEOUT or response.status in self.STATUS_CODES_SERVER_ERRORS:
            return []
        else:
            raise self._for_status(response, response.read())

    def _on_obj_ready(self, parent, operation, response):
        service_model = self.sqs_connection.meta.service_model
        if response.status == self.STATUS_CODE_OK:
            _, parsed = get_response(service_model.operation_model(operation), response.response)
            return parsed
        else:
            raise self._for_status(response, response.read())

    def _on_status_ready(self, parent, operation, response):
        service_model = self.sqs_connection.meta.service_model
        if response.status == self.STATUS_CODE_OK:
            httpres, _ = get_response(service_model.operation_model(operation), response.response)
            return httpres.code
        else:
            raise self._for_status(response, response.read())

    def _for_status(self, response, body):
        context = 'Empty body' if not body else 'HTTP Error'
        return Exception('Request {}  HTTP {}  {} ({})'.format(context, response.status, response.reason, body))