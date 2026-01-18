from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
@staticmethod
def _is_same_value(left, right):
    """check if two values are the same (stripping method wrappers, etc)"""
    return get_method_function(left) == get_method_function(right)