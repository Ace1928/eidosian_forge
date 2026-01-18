import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _is_retryable_exception(self, exception):
    return isinstance(exception, tuple(RETRYABLE_EXCEPTIONS['GENERAL_CONNECTION_ERROR']))