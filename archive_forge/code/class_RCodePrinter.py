from __future__ import annotations
from typing import Any
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = '_rcode'
    language = 'R'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 15, 'user_functions': {}, 'human': True, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_'}
    _operators = {'and': '&', 'or': '|', 'not': '!'}
    _relationals: dict[str, str] = {}

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
        return '// {}'.format(text)

    def _declare_number_const(self, name, value):
        return '{} = {};'.format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists of codelines
        """
        open_lines = []
        close_lines = []
        loopstart = 'for (%(var)s in %(start)s:%(end)s){'
        for i in indices:
            open_lines.append(loopstart % {'var': self._print(i.label), 'start': self._print(i.lower + 1), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_Pow(self, expr):
        if 'Pow' in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))

    def _print_Rational(self, expr):
        p, q = (int(expr.p), int(expr.q))
        return '%d.0/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        inds = [self._print(i) for i in expr.indices]
        return '%s[%s]' % (self._print(expr.base.label), ', '.join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return 'exp(1)'

    def _print_Pi(self, expr):
        return 'pi'

    def _print_Infinity(self, expr):
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        return '-Inf'

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(lhs, MatrixSymbol):
            lines = []
            for i, j in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return '\n'.join(lines)
        elif self._settings['contract'] and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond == True:
            last_line = '%s' % self._print(expr.args[-1].expr)
        else:
            last_line = 'ifelse(%s,%s,NA)' % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code = last_line
        for e, c in reversed(expr.args[:-1]):
            code = 'ifelse(%s,%s,' % (self._print(c), self._print(e)) + code + ')'
        return code

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        return '{}[{}]'.format(self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True), expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return '{} {} {};'.format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError('Only iterable currently supported is Range')
        body = self._print(expr.body)
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start, stop=stop - 1, step=step, body=body)

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)
        tab = '   '
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