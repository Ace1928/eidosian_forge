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
class ProxyProperty(object):
    """helper that proxies another attribute"""

    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, cls):
        if obj is None:
            cls = obj
        return getattr(obj, self.attr)

    def __set__(self, obj, value):
        setattr(obj, self.attr, value)

    def __delete__(self, obj):
        delattr(obj, self.attr)