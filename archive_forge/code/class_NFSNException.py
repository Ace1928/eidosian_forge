import time
import random
import string
import hashlib
from libcloud.utils.py3 import httplib, urlencode, basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
class NFSNException(ProviderError):

    def __init__(self, value, http_code, code, driver=None):
        self.code = code
        super().__init__(value, http_code, driver)