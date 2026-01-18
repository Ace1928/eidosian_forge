from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _obj_to_readable_str(obj):
    if isinstance(obj, str):
        return obj
    elif sys.version_info >= (3,) and isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return repr(obj)
    elif sys.version_info < (3,) and isinstance(obj, unicode):
        try:
            return obj.encode('ascii')
        except UnicodeEncodeError:
            return repr(obj)
    else:
        return repr(obj)