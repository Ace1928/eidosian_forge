import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def _signalmethod(func):

    @functools.wraps(func)
    def meth(*args, **kwargs):
        return func(*args, **kwargs)
    return meth