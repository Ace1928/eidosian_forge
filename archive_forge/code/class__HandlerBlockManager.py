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
class _HandlerBlockManager(object):

    def __init__(self, obj, handler_id):
        self.obj = obj
        self.handler_id = handler_id

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        GObjectModule.signal_handler_unblock(self.obj, self.handler_id)