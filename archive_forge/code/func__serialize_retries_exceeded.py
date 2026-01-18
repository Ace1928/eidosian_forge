import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_retries_exceeded(self, retries_exceeded, event_dict, **kwargs):
    event_dict['MaxRetriesExceeded'] = 1 if retries_exceeded else 0