from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _update_classmethod(self, oldcm, newcm):
    """Update a classmethod update."""
    self._update(None, None, oldcm.__get__(0), newcm.__get__(0))