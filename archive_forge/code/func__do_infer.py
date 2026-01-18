from types import ModuleType
import weakref
from numba.core.errors import ConstantInferenceError, NumbaError
from numba.core import ir
def _do_infer(self, name):
    if not isinstance(name, str):
        raise TypeError('infer_constant() called with non-str %r' % (name,))
    try:
        defn = self._func_ir.get_definition(name)
    except KeyError:
        raise ConstantInferenceError('no single definition for %r' % (name,))
    try:
        const = defn.infer_constant()
    except ConstantInferenceError:
        if isinstance(defn, ir.Expr):
            return self._infer_expr(defn)
        self._fail(defn)
    return const