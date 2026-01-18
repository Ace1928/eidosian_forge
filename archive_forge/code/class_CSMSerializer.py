import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
class CSMSerializer:
    _MAX_CLIENT_ID_LENGTH = 255
    _MAX_EXCEPTION_CLASS_LENGTH = 128
    _MAX_ERROR_CODE_LENGTH = 128
    _MAX_USER_AGENT_LENGTH = 256
    _MAX_MESSAGE_LENGTH = 512
    _RESPONSE_HEADERS_TO_EVENT_ENTRIES = {'x-amzn-requestid': 'XAmznRequestId', 'x-amz-request-id': 'XAmzRequestId', 'x-amz-id-2': 'XAmzId2'}
    _AUTH_REGEXS = {'v4': re.compile('AWS4-HMAC-SHA256 Credential=(?P<access_key>\\w+)/\\d+/(?P<signing_region>[a-z0-9-]+)/'), 's3': re.compile('AWS (?P<access_key>\\w+):')}
    _SERIALIZEABLE_EVENT_PROPERTIES = ['service', 'operation', 'timestamp', 'attempts', 'latency', 'retries_exceeded', 'url', 'request_headers', 'http_status_code', 'response_headers', 'parsed_error', 'wire_exception']

    def __init__(self, csm_client_id):
        """Serializes monitor events to CSM (Client Side Monitoring) format

        :type csm_client_id: str
        :param csm_client_id: The application identifier to associate
            to the serialized events
        """
        self._validate_client_id(csm_client_id)
        self.csm_client_id = csm_client_id

    def _validate_client_id(self, csm_client_id):
        if len(csm_client_id) > self._MAX_CLIENT_ID_LENGTH:
            raise ValueError(f'The value provided for csm_client_id: {csm_client_id} exceeds the maximum length of {self._MAX_CLIENT_ID_LENGTH} characters')

    def serialize(self, event):
        """Serializes a monitor event to the CSM format

        :type event: BaseMonitorEvent
        :param event: The event to serialize to bytes

        :rtype: bytes
        :returns: The CSM serialized form of the event
        """
        event_dict = self._get_base_event_dict(event)
        event_type = self._get_event_type(event)
        event_dict['Type'] = event_type
        for attr in self._SERIALIZEABLE_EVENT_PROPERTIES:
            value = getattr(event, attr, None)
            if value is not None:
                getattr(self, '_serialize_' + attr)(value, event_dict, event_type=event_type)
        return ensure_bytes(json.dumps(event_dict, separators=(',', ':')))

    def _get_base_event_dict(self, event):
        return {'Version': 1, 'ClientId': self.csm_client_id}

    def _serialize_service(self, service, event_dict, **kwargs):
        event_dict['Service'] = service

    def _serialize_operation(self, operation, event_dict, **kwargs):
        event_dict['Api'] = operation

    def _serialize_timestamp(self, timestamp, event_dict, **kwargs):
        event_dict['Timestamp'] = timestamp

    def _serialize_attempts(self, attempts, event_dict, **kwargs):
        event_dict['AttemptCount'] = len(attempts)
        if attempts:
            self._add_fields_from_last_attempt(event_dict, attempts[-1])

    def _add_fields_from_last_attempt(self, event_dict, last_attempt):
        if last_attempt.request_headers:
            region = self._get_region(last_attempt.request_headers)
            if region is not None:
                event_dict['Region'] = region
            event_dict['UserAgent'] = self._get_user_agent(last_attempt.request_headers)
        if last_attempt.http_status_code is not None:
            event_dict['FinalHttpStatusCode'] = last_attempt.http_status_code
        if last_attempt.parsed_error is not None:
            self._serialize_parsed_error(last_attempt.parsed_error, event_dict, 'ApiCall')
        if last_attempt.wire_exception is not None:
            self._serialize_wire_exception(last_attempt.wire_exception, event_dict, 'ApiCall')

    def _serialize_latency(self, latency, event_dict, event_type):
        if event_type == 'ApiCall':
            event_dict['Latency'] = latency
        elif event_type == 'ApiCallAttempt':
            event_dict['AttemptLatency'] = latency

    def _serialize_retries_exceeded(self, retries_exceeded, event_dict, **kwargs):
        event_dict['MaxRetriesExceeded'] = 1 if retries_exceeded else 0

    def _serialize_url(self, url, event_dict, **kwargs):
        event_dict['Fqdn'] = urlparse(url).netloc

    def _serialize_request_headers(self, request_headers, event_dict, **kwargs):
        event_dict['UserAgent'] = self._get_user_agent(request_headers)
        if self._is_signed(request_headers):
            event_dict['AccessKey'] = self._get_access_key(request_headers)
        region = self._get_region(request_headers)
        if region is not None:
            event_dict['Region'] = region
        if 'X-Amz-Security-Token' in request_headers:
            event_dict['SessionToken'] = request_headers['X-Amz-Security-Token']

    def _serialize_http_status_code(self, http_status_code, event_dict, **kwargs):
        event_dict['HttpStatusCode'] = http_status_code

    def _serialize_response_headers(self, response_headers, event_dict, **kwargs):
        for header, entry in self._RESPONSE_HEADERS_TO_EVENT_ENTRIES.items():
            if header in response_headers:
                event_dict[entry] = response_headers[header]

    def _serialize_parsed_error(self, parsed_error, event_dict, event_type, **kwargs):
        field_prefix = 'Final' if event_type == 'ApiCall' else ''
        event_dict[field_prefix + 'AwsException'] = self._truncate(parsed_error['Code'], self._MAX_ERROR_CODE_LENGTH)
        event_dict[field_prefix + 'AwsExceptionMessage'] = self._truncate(parsed_error['Message'], self._MAX_MESSAGE_LENGTH)

    def _serialize_wire_exception(self, wire_exception, event_dict, event_type, **kwargs):
        field_prefix = 'Final' if event_type == 'ApiCall' else ''
        event_dict[field_prefix + 'SdkException'] = self._truncate(wire_exception.__class__.__name__, self._MAX_EXCEPTION_CLASS_LENGTH)
        event_dict[field_prefix + 'SdkExceptionMessage'] = self._truncate(str(wire_exception), self._MAX_MESSAGE_LENGTH)

    def _get_event_type(self, event):
        if isinstance(event, APICallEvent):
            return 'ApiCall'
        elif isinstance(event, APICallAttemptEvent):
            return 'ApiCallAttempt'

    def _get_access_key(self, request_headers):
        auth_val = self._get_auth_value(request_headers)
        _, auth_match = self._get_auth_match(auth_val)
        return auth_match.group('access_key')

    def _get_region(self, request_headers):
        if not self._is_signed(request_headers):
            return None
        auth_val = self._get_auth_value(request_headers)
        signature_version, auth_match = self._get_auth_match(auth_val)
        if signature_version != 'v4':
            return None
        return auth_match.group('signing_region')

    def _get_user_agent(self, request_headers):
        return self._truncate(ensure_unicode(request_headers.get('User-Agent', '')), self._MAX_USER_AGENT_LENGTH)

    def _is_signed(self, request_headers):
        return 'Authorization' in request_headers

    def _get_auth_value(self, request_headers):
        return ensure_unicode(request_headers['Authorization'])

    def _get_auth_match(self, auth_val):
        for signature_version, regex in self._AUTH_REGEXS.items():
            match = regex.match(auth_val)
            if match:
                return (signature_version, match)
        return (None, None)

    def _truncate(self, text, max_length):
        if len(text) > max_length:
            logger.debug('Truncating following value to maximum length of %s: %s', text, max_length)
            return text[:max_length]
        return text