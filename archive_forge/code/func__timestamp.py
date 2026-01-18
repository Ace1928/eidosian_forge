import time
import random
import string
import hashlib
from libcloud.utils.py3 import httplib, urlencode, basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
def _timestamp(self):
    """Return the current number of seconds since the Unix epoch,
        as a string."""
    return str(int(time.time()))