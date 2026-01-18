import os
import re
import time
import hashlib
import threading
import warnings
from base64 import b64encode
from .compat import urlparse, str, basestring
from .cookies import extract_cookies_to_jar
from ._internal_utils import to_native_string
from .utils import parse_dict_header
def _basic_auth_str(username, password):
    """Returns a Basic Auth string."""
    if not isinstance(username, basestring):
        warnings.warn("Non-string usernames will no longer be supported in Requests 3.0.0. Please convert the object you've passed in ({!r}) to a string or bytes object in the near future to avoid problems.".format(username), category=DeprecationWarning)
        username = str(username)
    if not isinstance(password, basestring):
        warnings.warn("Non-string passwords will no longer be supported in Requests 3.0.0. Please convert the object you've passed in ({!r}) to a string or bytes object in the near future to avoid problems.".format(type(password)), category=DeprecationWarning)
        password = str(password)
    if isinstance(username, str):
        username = username.encode('latin1')
    if isinstance(password, str):
        password = password.encode('latin1')
    authstr = 'Basic ' + to_native_string(b64encode(b':'.join((username, password))).strip())
    return authstr