import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
@staticmethod
def _is_valid_close_status(code):
    return code in VALID_CLOSE_STATUS or 3000 <= code < 5000