from .pycode import (
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key
class NumExprPrinter(LambdaPrinter):
    printmethod = '_numexprcode'
    _numexpr_functions = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'atan2': 'arctan2', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh', 'ln': 'log', 'log': 'log', 'exp': 'exp', 'sqrt': 'sqrt', 'Abs': 'abs', 'conjugate': 'conj', 'im': 'imag', 're': 'real', 'where': 'where', 'complex': 'complex', 'contains': 'contains'}
    module = 'numexpr'

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_seq(self, seq, delimiter=', '):
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ''

    def _print_Function(self, e):
        func_name = e.func.__name__
        nstr = self._numexpr_functions.get(func_name, None)
        if nstr is None:
            if hasattr(e, '_imp_'):
                return '(%s)' % self._print(e._imp_(*e.args))
            else:
                raise TypeError("numexpr does not support function '%s'" % func_name)
        return '%s(%s)' % (nstr, self._print_seq(e.args))

    def _print_Piecewise(self, expr):
        """Piecewise function printer"""
        exprs = [self._print(arg.expr) for arg in expr.args]
        conds = [self._print(arg.cond) for arg in expr.args]
        ans = []
        parenthesis_count = 0
        is_last_cond_True = False
        for cond, expr in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))
                parenthesis_count += 1
        if not is_last_cond_True:
            ans.append('log(-1)')
        return ''.join(ans) + ')' * parenthesis_count

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def blacklisted(self, expr):
        raise TypeError('numexpr cannot be used with %s' % expr.__class__.__name__)
    _print_SparseRepMatrix = _print_MutableSparseMatrix = _print_ImmutableSparseMatrix = _print_Matrix = _print_DenseMatrix = _print_MutableDenseMatrix = _print_ImmutableMatrix = _print_ImmutableDenseMatrix = blacklisted
    _print_list = _print_tuple = _print_Tuple = _print_dict = _print_Dict = blacklisted

    def _print_NumExprEvaluate(self, expr):
        evaluate = self._module_format(self.module + '.evaluate')
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    def doprint(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    def _print_Return(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        r, = expr.args
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    def _print_Assignment(self, expr):
        from sympy.codegen.pynodes import NumExprEvaluate
        lhs, rhs, *args = expr.args
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    def _print_CodeBlock(self, expr):
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        args = [arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args]
        return super()._print_CodeBlock(self, expr.func(*args))