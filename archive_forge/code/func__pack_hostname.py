import hashlib
import hmac
import os
import six
from ._cookiejar import SimpleCookieJar
from ._exceptions import *
from ._http import *
from ._logging import *
from ._socket import *
def _pack_hostname(hostname):
    if ':' in hostname:
        return '[' + hostname + ']'
    return hostname