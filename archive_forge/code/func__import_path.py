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
def _import_path(self, path):
    """retrieve obj and final attribute name from resource path"""
    name, attr = path.split(':')
    obj = __import__(name, fromlist=[attr], level=0)
    while '.' in attr:
        head, attr = attr.split('.', 1)
        obj = getattr(obj, head)
    return (obj, attr)