from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
class RustCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of Rust code"""
    printmethod = '_rust_code'
    language = 'Rust'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_', 'inline': False}

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        return p * 5

    def _get_statement(self, codestring):
        return '%s;' % codestring

    def _get_comment(self, text):
        return '// %s' % text

    def _declare_number_const(self, name, value):
        return 'const %s: f64 = %s;' % (name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = 'for %(var)s in %(start)s..%(end)s {'
        for i in indices:
            open_lines.append(loopstart % {'var': self._print(i), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_caller_var(self, expr):
        if len(expr.args) > 1:
            return '(' + self._print(expr) + ')'
        elif expr.is_number:
            return self._print(expr, _type=True)
        else:
            return self._print(expr)

    def _print_Function(self, expr):
        """
        basic function for printing `Function`

        Function Style :

        1. args[0].func(args[1:]), method with arguments
        2. args[0].func(), method without arguments
        3. args[1].func(), method without arguments (e.g. (e, x) => x.exp())
        4. func(args), function with arguments
        """
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            func = None
            style = 1
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func, style in cond_func:
                    if cond(*expr.args):
                        break
            if func is not None:
                if style == 1:
                    ret = '%(var)s.%(method)s(%(args)s)' % {'var': self._print_caller_var(expr.args[0]), 'method': func, 'args': self.stringify(expr.args[1:], ', ') if len(expr.args) > 1 else ''}
                elif style == 2:
                    ret = '%(var)s.%(method)s()' % {'var': self._print_caller_var(expr.args[0]), 'method': func}
                elif style == 3:
                    ret = '%(var)s.%(method)s()' % {'var': self._print_caller_var(expr.args[1]), 'method': func}
                else:
                    ret = '%(func)s(%(args)s)' % {'func': func, 'args': self.stringify(expr.args, ', ')}
                return ret
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all((self._can_print(f) for f in required_fs)):
                return self._print(expr.rewrite(target_f))
        else:
            return self._print_not_supported(expr)

    def _print_Pow(self, expr):
        if expr.base.is_integer and (not expr.exp.is_integer):
            expr = type(expr)(Float(expr.base), expr.exp)
            return self._print(expr)
        return self._print_Function(expr)

    def _print_Float(self, expr, _type=False):
        ret = super()._print_Float(expr)
        if _type:
            return ret + '_f64'
        else:
            return ret

    def _print_Integer(self, expr, _type=False):
        ret = super()._print_Integer(expr)
        if _type:
            return ret + '_i32'
        else:
            return ret

    def _print_Rational(self, expr):
        p, q = (int(expr.p), int(expr.q))
        return '%d_f64/%d.0' % (p, q)

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Indexed(self, expr):
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i] * offset
            offset *= dims[i]
        return '%s[%s]' % (self._print(expr.base.label), self._print(elem))

    def _print_Idx(self, expr):
        return expr.label.name

    def _print_Dummy(self, expr):
        return expr.name

    def _print_Exp1(self, expr, _type=False):
        return 'E'

    def _print_Pi(self, expr, _type=False):
        return 'PI'

    def _print_Infinity(self, expr, _type=False):
        return 'INFINITY'

    def _print_NegativeInfinity(self, expr, _type=False):
        return 'NEG_INFINITY'

    def _print_BooleanTrue(self, expr, _type=False):
        return 'true'

    def _print_BooleanFalse(self, expr, _type=False):
        return 'false'

    def _print_bool(self, expr, _type=False):
        return str(expr).lower()

    def _print_NaN(self, expr, _type=False):
        return 'NAN'

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        for i, (e, c) in enumerate(expr.args):
            if i == 0:
                lines.append('if (%s) {' % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines[-1] += ' else {'
            else:
                lines[-1] += ' else if (%s) {' % self._print(c)
            code0 = self._print(e)
            lines.append(code0)
            lines.append('}')
        if self._settings['inline']:
            return ' '.join(lines)
        else:
            return '\n'.join(lines)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixBase(self, A):
        if A.cols == 1:
            return '[%s]' % ', '.join((self._print(a) for a in A))
        else:
            raise ValueError('Full Matrix Support in Rust need Crates (https://crates.io/keywords/matrix).')

    def _print_SparseRepMatrix(self, mat):
        return self._print_not_supported(mat)

    def _print_MatrixElement(self, expr):
        return '%s[%s]' % (expr.parent, expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*%s)' % name
        else:
            return name

    def _print_Assignment(self, expr):
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '    '
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')
        code = [line.lstrip(' \t') for line in code]
        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]
        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append('%s%s' % (tab * level, line))
            level += increase[n]
        return pretty