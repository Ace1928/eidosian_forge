import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _handle_after_call_error(self, context, exception, **kwargs):
    context['current_api_call_event'].retries_exceeded = self._is_retryable_exception(exception)
    return self._complete_api_call(context)