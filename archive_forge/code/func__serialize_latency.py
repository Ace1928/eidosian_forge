import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_latency(self, latency, event_dict, event_type):
    if event_type == 'ApiCall':
        event_dict['Latency'] = latency
    elif event_type == 'ApiCallAttempt':
        event_dict['AttemptLatency'] = latency