import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _get_event_type(self, event):
    if isinstance(event, APICallEvent):
        return 'ApiCall'
    elif isinstance(event, APICallAttemptEvent):
        return 'ApiCallAttempt'