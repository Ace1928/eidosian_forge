from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
class JuliaCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Julia code.
    """
    printmethod = '_julia'
    language = 'Julia'
    _operators = {'and': '&&', 'or': '||', 'not': '!'}
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'inline': True}

    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p * 5

    def _get_statement(self, codestring):
        return '%s' % codestring

    def _get_comment(self, text):
        return '# {}'.format(text)

    def _declare_number_const(self, name, value):
        return 'const {} = {}'.format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            var, start, stop = map(self._print, [i.label, i.lower + 1, i.upper + 1])
            open_lines.append('for %s = %s:%s' % (var, start, stop))
            close_lines.append('end')
        return (open_lines, close_lines)

    def _print_Mul(self, expr):
        if expr.is_number and expr.is_imaginary and expr.as_coeff_Mul()[0].is_integer:
            return '%sim' % self._print(-S.ImaginaryUnit * expr)
        prec = precedence(expr)
        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = '-'
        else:
            sign = ''
        a = []
        b = []
        pow_paren = []
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity and (item.p == 1):
                b.append(Rational(item.q))
            else:
                a.append(item)
        a = a or [S.One]
        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]

        def multjoin(a, a_str):
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i - 1].is_number else '.*'
                r = '%s %s %s' % (r, mulsym, a_str[i])
            return r
        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return '%s %s %s' % (sign + multjoin(a, a_str), divsym, b_str[0])
        else:
            divsym = '/' if all((bi.is_number for bi in b)) else './'
            return '%s %s (%s)' % (sign + multjoin(a, a_str), divsym, multjoin(b, b_str))

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        powsymbol = '^' if all((x.is_number for x in expr.args)) else '.^'
        PREC = precedence(expr)
        if equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return '1 %s sqrt(%s)' % (sym, self._print(expr.base))
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return '1 %s %s' % (sym, self.parenthesize(expr.base, PREC))
        return '%s %s %s' % (self.parenthesize(expr.base, PREC), powsymbol, self.parenthesize(expr.exp, PREC))

    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s ^ %s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))

    def _print_Pi(self, expr):
        if self._settings['inline']:
            return 'pi'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_ImaginaryUnit(self, expr):
        return 'im'

    def _print_Exp1(self, expr):
        if self._settings['inline']:
            return 'e'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_EulerGamma(self, expr):
        if self._settings['inline']:
            return 'eulergamma'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_Catalan(self, expr):
        if self._settings['inline']:
            return 'catalan'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_GoldenRatio(self, expr):
        if self._settings['inline']:
            return 'golden'
        else:
            return super()._print_NumberSymbol(expr)

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if not self._settings['inline'] and isinstance(expr.rhs, Piecewise):
            expressions = []
            conditions = []
            for e, c in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_Infinity(self, expr):
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        return '-Inf'

    def _print_NaN(self, expr):
        return 'NaN'

    def _print_list(self, expr):
        return 'Any[' + ', '.join((self._print(a) for a in expr)) + ']'

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return '(%s,)' % self._print(expr[0])
        else:
            return '(%s)' % self.stringify(expr, ', ')
    _print_Tuple = _print_tuple

    def _print_BooleanTrue(self, expr):
        return 'true'

    def _print_BooleanFalse(self, expr):
        return 'false'

    def _print_bool(self, expr):
        return str(expr).lower()

    def _print_MatrixBase(self, A):
        if S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            return '[%s]' % A[0, 0]
        elif A.rows == 1:
            return '[%s]' % A.table(self, rowstart='', rowend='', colsep=' ')
        elif A.cols == 1:
            return '[%s]' % ', '.join([self._print(a) for a in A])
        return '[%s]' % A.table(self, rowstart='', rowend='', rowsep=';\n', colsep=' ')

    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        I = Matrix([k[0] + 1 for k in L])
        J = Matrix([k[1] + 1 for k in L])
        AIJ = Matrix([k[2] for k in L])
        return 'sparse(%s, %s, %s, %s, %s)' % (self._print(I), self._print(J), self._print(AIJ), A.rows, A.cols)

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '[%s,%s]' % (expr.i + 1, expr.j + 1)

    def _print_MatrixSlice(self, expr):

        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return self._print(expr.parent) + '[' + strslice(expr.rowslice, expr.parent.shape[0]) + ',' + strslice(expr.colslice, expr.parent.shape[1]) + ']'

    def _print_Indexed(self, expr):
        inds = [self._print(i) for i in expr.indices]
        return '%s[%s]' % (self._print(expr.base.label), ','.join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Identity(self, expr):
        return 'eye(%s)' % self._print(expr.shape[0])

    def _print_HadamardProduct(self, expr):
        return ' .* '.join([self.parenthesize(arg, precedence(expr)) for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC)])

    def _print_Rational(self, expr):
        if expr.q == 1:
            return str(expr.p)
        return '%s // %s' % (expr.p, expr.q)

    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi / (2 * x)) * besselj(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi / (2 * x)) * bessely(expr.order + S.Half, x)
        return self._print(expr2)

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        if self._settings['inline']:
            ecpairs = ['({}) ? ({}) :'.format(self._print(c), self._print(e)) for e, c in expr.args[:-1]]
            elast = ' (%s)' % self._print(expr.args[-1].expr)
            pw = '\n'.join(ecpairs) + elast
            return '(' + pw + ')'
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append('if (%s)' % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append('else')
                else:
                    lines.append('elseif (%s)' % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append('end')
            return '\n'.join(lines)

    def _print_MatMul(self, expr):
        c, m = expr.as_coeff_mmul()
        sign = ''
        if c.is_number:
            re, im = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
        return sign + ' * '.join((self.parenthesize(arg, precedence(expr)) for arg in expr.args))

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '    '
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')
        code = [line.lstrip(' \t') for line in code]
        increase = [int(any((search(re, line) for re in inc_regex))) for line in code]
        decrease = [int(any((search(re, line) for re in dec_regex))) for line in code]
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