import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
class Idle(Source):

    def __new__(cls, priority=GLib.PRIORITY_DEFAULT):
        source = GLib.idle_source_new()
        source.__class__ = cls
        return source

    def __init__(self, priority=GLib.PRIORITY_DEFAULT):
        super(Source, self).__init__()
        if priority != GLib.PRIORITY_DEFAULT:
            self.set_priority(priority)