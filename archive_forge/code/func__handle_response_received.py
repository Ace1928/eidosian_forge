import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _handle_response_received(self, parsed_response, context, exception, **kwargs):
    attempt_event = context.pop('current_api_call_attempt_event')
    attempt_event.latency = self._get_latency(attempt_event)
    if parsed_response is not None:
        attempt_event.http_status_code = parsed_response['ResponseMetadata']['HTTPStatusCode']
        attempt_event.response_headers = parsed_response['ResponseMetadata']['HTTPHeaders']
        attempt_event.parsed_error = parsed_response.get('Error')
    else:
        attempt_event.wire_exception = exception
    return attempt_event