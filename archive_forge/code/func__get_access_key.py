import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _get_access_key(self, request_headers):
    auth_val = self._get_auth_value(request_headers)
    _, auth_match = self._get_auth_match(auth_val)
    return auth_match.group('access_key')