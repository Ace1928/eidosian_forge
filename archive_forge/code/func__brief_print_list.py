import os
import sys
import hashlib
import uuid
import warnings
import collections
import weakref
import requests
import numpy as np
from .. import ndarray
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit // 2], limit) + ', ..., ' + _brief_print_list(lst[-limit // 2:], limit)
    return ', '.join(["'%s'" % str(i) for i in lst])