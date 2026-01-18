from types import ModuleType
import weakref
from numba.core.errors import ConstantInferenceError, NumbaError
from numba.core import ir
def _infer_getattr(self, value, expr):
    if isinstance(value, (ModuleType, type)):
        try:
            return getattr(value, expr.attr)
        except AttributeError:
            pass
    self._fail(expr)