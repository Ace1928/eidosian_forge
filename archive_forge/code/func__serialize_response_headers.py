import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_response_headers(self, response_headers, event_dict, **kwargs):
    for header, entry in self._RESPONSE_HEADERS_TO_EVENT_ENTRIES.items():
        if header in response_headers:
            event_dict[entry] = response_headers[header]