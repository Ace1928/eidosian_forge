import hashlib
import hmac
import os
import six
from ._cookiejar import SimpleCookieJar
from ._exceptions import *
from ._http import *
from ._logging import *
from ._socket import *
def _create_sec_websocket_key():
    randomness = os.urandom(16)
    return base64encode(randomness).decode('utf-8').strip()