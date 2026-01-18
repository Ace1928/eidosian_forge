import sys
import warnings
from gi.repository import GObject
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from .._gtktemplate import Template, _extract_handler_and_args
from ..overrides import (override, strip_boolean_result, deprecated_init,
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning
def _builder_connect_callback(builder, gobj, signal_name, handler_name, connect_obj, flags, obj_or_map):
    handler, args = _extract_handler_and_args(obj_or_map, handler_name)
    after = flags & GObject.ConnectFlags.AFTER
    if connect_obj is not None:
        if after:
            gobj.connect_object_after(signal_name, handler, connect_obj, *args)
        else:
            gobj.connect_object(signal_name, handler, connect_obj, *args)
    elif after:
        gobj.connect_after(signal_name, handler, *args)
    else:
        gobj.connect(signal_name, handler, *args)