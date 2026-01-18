import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
class LLVMJitPrinter(Printer):
    """Convert expressions to LLVM IR"""

    def __init__(self, module, builder, fn, *args, **kwargs):
        self.func_arg_map = kwargs.pop('func_arg_map', {})
        if not llvmlite:
            raise ImportError('llvmlite is required for LLVMJITPrinter')
        super().__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _print_Number(self, n):
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            val = self.func_arg_map.get(s)
        if not val:
            raise LookupError('Symbol not found: %s' % s)
        return val

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get('sqrt')
            if not fn:
                fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
                fn = ll.Function(self.module, fn_type, 'sqrt')
                self.ext_fn['sqrt'] = fn
            return self.builder.call(fn, [base0], 'sqrt')
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)
        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get('pow')
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ll.Function(self.module, fn_type, 'pow')
            self.ext_fn['pow'] = fn
        return self.builder.call(fn, [base0, exp0], 'pow')

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            fn = ll.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def emptyPrinter(self, expr):
        raise TypeError('Unsupported type for LLVM JIT conversion: %s' % type(expr))