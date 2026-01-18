import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
def _recently_failed(self, cache_key):
    if cache_key in self._failed_attempts:
        now = self._time()
        if now < self._failed_attempts[cache_key]:
            return True
        del self._failed_attempts[cache_key]
    return False