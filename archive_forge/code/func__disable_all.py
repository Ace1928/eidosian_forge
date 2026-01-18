import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def _disable_all():
    """Reverse all effects of the enable_xxx() calls except for
    require_version() calls and imports.
    """
    _enabled_registry.clear()
    for obj, name, old_value in reversed(_patches):
        if old_value is _unset:
            delattr(obj, name)
        else:
            delattr(obj, name)
            if getattr(obj, name, _unset) is not old_value:
                setattr(obj, name, old_value)
    del _patches[:]
    for name, old_value in reversed(_module_patches):
        if old_value is _unset:
            del sys.modules[name]
        else:
            sys.modules[name] = old_value
    del _module_patches[:]