import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _complete_api_call(self, context):
    call_event = context.pop('current_api_call_event')
    call_event.latency = self._get_latency(call_event)
    return call_event