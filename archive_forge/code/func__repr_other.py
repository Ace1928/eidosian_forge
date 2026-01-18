import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_other(self, obj, level):
    return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)