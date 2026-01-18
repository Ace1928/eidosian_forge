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
class AuthBase(object):
    """Base class that all auth implementations derive from"""

    def __call__(self, r):
        raise NotImplementedError('Auth hooks must be callable.')