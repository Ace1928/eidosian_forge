from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _update_method(self, oldmeth, newmeth):
    """Update a method object."""
    if hasattr(oldmeth, 'im_func') and hasattr(newmeth, 'im_func'):
        self._update(None, None, oldmeth.im_func, newmeth.im_func)
    elif hasattr(oldmeth, '__func__') and hasattr(newmeth, '__func__'):
        self._update(None, None, oldmeth.__func__, newmeth.__func__)
    return oldmeth