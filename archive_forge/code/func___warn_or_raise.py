import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def __warn_or_raise(text, warn_class, to_raise, raise_reason):
    if _should_raise:
        raise to_raise(raise_reason)
    else:
        warnings.warn(text, warn_class, stacklevel=2)