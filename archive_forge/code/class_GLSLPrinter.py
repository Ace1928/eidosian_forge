from __future__ import annotations
from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce
class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported: set[Basic] = set()
    printmethod = '_glsl'
    language = 'GLSL'
    _default_settings = {'use_operators': True, 'zero': 0, 'mat_nested': False, 'mat_separator': ',\n', 'mat_transpose': False, 'array_type': 'float', 'glsl_types': True, 'order': None, 'full_prec': 'auto', 'precision': 9, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'error_on_reserved': False, 'reserved_word_suffix': '_'}

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p * 5

    def _get_statement(self, codestring):
        return '%s;' % codestring

    def _get_comment(self, text):
        return '// {}'.format(text)

    def _declare_number_const(self, name, value):
        return 'float {} = {};'.format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

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

    def _print_MatrixBase(self, mat):
        mat_separator = self._settings['mat_separator']
        mat_transpose = self._settings['mat_transpose']
        column_vector = mat.rows == 1 if mat_transpose else mat.cols == 1
        A = mat.transpose() if mat_transpose != column_vector else mat
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = A.cols * A.rows
        array_constructor = '{}[{}]'.format(array_type, array_size)
        if A.cols == 1:
            return self._print(A[0])
        if A.rows <= 4 and A.cols <= 4 and glsl_types:
            if A.rows == 1:
                return 'vec{}{}'.format(A.cols, A.table(self, rowstart='(', rowend=')'))
            elif A.rows == A.cols:
                return 'mat{}({})'.format(A.rows, A.table(self, rowsep=', ', rowstart='', rowend=''))
            else:
                return 'mat{}x{}({})'.format(A.cols, A.rows, A.table(self, rowsep=', ', rowstart='', rowend=''))
        elif S.One in A.shape:
            return '{}({})'.format(array_constructor, A.table(self, rowsep=mat_separator, rowstart='', rowend=''))
        elif not self._settings['mat_nested']:
            return '{}(\n{}\n) /* a {}x{} matrix */'.format(array_constructor, A.table(self, rowsep=mat_separator, rowstart='', rowend=''), A.rows, A.cols)
        elif self._settings['mat_nested']:
            return '{}[{}][{}](\n{}\n)'.format(array_type, A.rows, A.cols, A.table(self, rowsep=mat_separator, rowstart='float[](', rowend=')'))

    def _print_SparseRepMatrix(self, mat):
        return self._print_not_supported(mat)

    def _traverse_matrix_indices(self, mat):
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            rows, cols = mat.shape
        else:
            cols, rows = mat.shape
        return ((i, j) for i in range(cols) for j in range(rows))

    def _print_MatrixElement(self, expr):
        nest = self._settings['mat_nested']
        glsl_types = self._settings['glsl_types']
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            cols, rows = expr.parent.shape
            i, j = (expr.j, expr.i)
        else:
            rows, cols = expr.parent.shape
            i, j = (expr.i, expr.j)
        pnt = self._print(expr.parent)
        if glsl_types and (rows <= 4 and cols <= 4 or nest):
            return '{}[{}][{}]'.format(pnt, i, j)
        else:
            return '{}[{}]'.format(pnt, i + j * rows)

    def _print_list(self, expr):
        l = ', '.join((self._print(item) for item in expr))
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = len(expr)
        array_constructor = '{}[{}]'.format(array_type, array_size)
        if array_size <= 4 and glsl_types:
            return 'vec{}({})'.format(array_size, l)
        else:
            return '{}({})'.format(array_constructor, l)
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = 'for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){'
        for i in indices:
            open_lines.append(loopstart % {'varble': self._print(i.label), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
            close_lines.append('}')
        return (open_lines, close_lines)

    def _print_Function_with_args(self, func, func_args):
        if func in self.known_functions:
            cond_func = self.known_functions[func]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func in cond_func:
                    if cond(func_args):
                        break
            if func is not None:
                try:
                    return func(*[self.parenthesize(item, 0) for item in func_args])
                except TypeError:
                    return '{}({})'.format(func, self.stringify(func_args, ', '))
        elif isinstance(func, Lambda):
            return self._print(func(*func_args))
        else:
            return self._print_not_supported(func)

    def _print_Piecewise(self, expr):
        from sympy.codegen.ast import Assignment
        if expr.args[-1].cond != True:
            raise ValueError('All Piecewise expressions must contain an (expr, True) statement to be used as a default condition. Without one, the generated expression may not evaluate to anything under some condition.')
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append('if (%s) {' % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append('else {')
                else:
                    lines.append('else if (%s) {' % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append('}')
            return '\n'.join(lines)
        else:
            ecpairs = ['((%s) ? (\n%s\n)\n' % (self._print(c), self._print(e)) for e, c in expr.args[:-1]]
            last_line = ': (\n%s\n)' % self._print(expr.args[-1].expr)
            return ': '.join(ecpairs) + last_line + ' '.join([')' * len(ecpairs)])

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Indexed(self, expr):
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i] * offset
            offset *= dims[i]
        return '{}[{}]'.format(self._print(expr.base.label), self._print(elem))

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            try:
                e = self._print(float(expr.exp))
            except TypeError:
                e = self._print(expr.exp)
            return self._print_Function_with_args('pow', (self._print(expr.base), e))

    def _print_int(self, expr):
        return str(float(expr))

    def _print_Rational(self, expr):
        return '{}.0/{}.0'.format(expr.p, expr.q)

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Add(self, expr, order=None):
        if self._settings['use_operators']:
            return CodePrinter._print_Add(self, expr, order=order)
        terms = expr.as_ordered_terms()

        def partition(p, l):
            return reduce(lambda x, y: (x[0] + [y], x[1]) if p(y) else (x[0], x[1] + [y]), l, ([], []))

        def add(a, b):
            return self._print_Function_with_args('add', (a, b))
        neg, pos = partition(lambda arg: arg.could_extract_minus_sign(), terms)
        if pos:
            s = pos = reduce(lambda a, b: add(a, b), (self._print(t) for t in pos))
        else:
            s = pos = self._print(self._settings['zero'])
        if neg:
            neg = reduce(lambda a, b: add(a, b), (self._print(-n) for n in neg))
            s = self._print_Function_with_args('sub', (pos, neg))
        return s

    def _print_Mul(self, expr, **kwargs):
        if self._settings['use_operators']:
            return CodePrinter._print_Mul(self, expr, **kwargs)
        terms = expr.as_ordered_factors()

        def mul(a, b):
            return self._print_Function_with_args('mul', (a, b))
        s = reduce(lambda a, b: mul(a, b), (self._print(t) for t in terms))
        return s