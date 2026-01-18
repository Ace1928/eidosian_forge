from types import ModuleType
import weakref
from numba.core.errors import ConstantInferenceError, NumbaError
from numba.core import ir
class ConstantInference(object):
    """
    A constant inference engine for a given interpreter.
    Inference inspects the IR to try and compute a compile-time constant for
    a variable.

    This shouldn't be used directly, instead call Interpreter.infer_constant().
    """

    def __init__(self, func_ir):
        self._func_ir = weakref.proxy(func_ir)
        self._cache = {}

    def infer_constant(self, name, loc=None):
        """
        Infer a constant value for the given variable *name*.
        If no value can be inferred, numba.errors.ConstantInferenceError
        is raised.
        """
        if name not in self._cache:
            try:
                self._cache[name] = (True, self._do_infer(name))
            except ConstantInferenceError as exc:
                self._cache[name] = (False, (exc.__class__, exc.args))
        success, val = self._cache[name]
        if success:
            return val
        else:
            exc, args = val
            if issubclass(exc, NumbaError):
                raise exc(*args, loc=loc)
            else:
                raise exc(*args)

    def _fail(self, val):
        raise ConstantInferenceError('Constant inference not possible for: %s' % (val,), loc=None)

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

    def _infer_expr(self, expr):
        if expr.op == 'call':
            func = self.infer_constant(expr.func.name, loc=expr.loc)
            return self._infer_call(func, expr)
        elif expr.op == 'getattr':
            value = self.infer_constant(expr.value.name, loc=expr.loc)
            return self._infer_getattr(value, expr)
        elif expr.op == 'build_list':
            return [self.infer_constant(i.name, loc=expr.loc) for i in expr.items]
        elif expr.op == 'build_tuple':
            return tuple((self.infer_constant(i.name, loc=expr.loc) for i in expr.items))
        self._fail(expr)

    def _infer_call(self, func, expr):
        if expr.kws or expr.vararg:
            self._fail(expr)
        _slice = func in (slice,)
        _exc = isinstance(func, type) and issubclass(func, BaseException)
        if _slice or _exc:
            args = [self.infer_constant(a.name, loc=expr.loc) for a in expr.args]
            if _slice:
                return func(*args)
            elif _exc:
                return (func, args)
            else:
                assert 0, 'Unreachable'
        self._fail(expr)

    def _infer_getattr(self, value, expr):
        if isinstance(value, (ModuleType, type)):
            try:
                return getattr(value, expr.attr)
            except AttributeError:
                pass
        self._fail(expr)