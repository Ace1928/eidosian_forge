from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
class C89CodePrinter(CodePrinter):
    """A printer to convert Python expressions to strings of C code"""
    printmethod = '_ccode'
    language = 'C'
    standard = 'C89'
    reserved_words = set(reserved_words)
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 17, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False, 'contract': True, 'dereference': set(), 'error_on_reserved': False, 'reserved_word_suffix': '_'}
    type_aliases = {real: float64, complex_: complex128, integer: intc}
    type_mappings: dict[Type, Any] = {real: 'double', intc: 'int', float32: 'float', float64: 'double', integer: 'int', bool_: 'bool', int8: 'int8_t', int16: 'int16_t', int32: 'int32_t', int64: 'int64_t', uint8: 'int8_t', uint16: 'int16_t', uint32: 'int32_t', uint64: 'int64_t'}
    type_headers = {bool_: {'stdbool.h'}, int8: {'stdint.h'}, int16: {'stdint.h'}, int32: {'stdint.h'}, int64: {'stdint.h'}, uint8: {'stdint.h'}, uint16: {'stdint.h'}, uint32: {'stdint.h'}, uint64: {'stdint.h'}}
    type_macros: dict[Type, tuple[str, ...]] = {}
    type_func_suffixes = {float32: 'f', float64: '', float80: 'l'}
    type_literal_suffixes = {float32: 'F', float64: '', float80: 'L'}
    type_math_macro_suffixes = {float80: 'l'}
    math_macros = None
    _ns = ''
    _kf: dict[str, Any] = known_functions_C89

    def __init__(self, settings=None):
        settings = settings or {}
        if self.math_macros is None:
            self.math_macros = settings.pop('math_macros', get_math_macros())
        self.type_aliases = dict(chain(self.type_aliases.items(), settings.pop('type_aliases', {}).items()))
        self.type_mappings = dict(chain(self.type_mappings.items(), settings.pop('type_mappings', {}).items()))
        self.type_headers = dict(chain(self.type_headers.items(), settings.pop('type_headers', {}).items()))
        self.type_macros = dict(chain(self.type_macros.items(), settings.pop('type_macros', {}).items()))
        self.type_func_suffixes = dict(chain(self.type_func_suffixes.items(), settings.pop('type_func_suffixes', {}).items()))
        self.type_literal_suffixes = dict(chain(self.type_literal_suffixes.items(), settings.pop('type_literal_suffixes', {}).items()))
        self.type_math_macro_suffixes = dict(chain(self.type_math_macro_suffixes.items(), settings.pop('type_math_macro_suffixes', {}).items()))
        super().__init__(settings)
        self.known_functions = dict(self._kf, **settings.get('user_functions', {}))
        self._dereference = set(settings.get('dereference', []))
        self.headers = set()
        self.libraries = set()
        self.macros = set()

    def _rate_index_position(self, p):
        return p * 5

    def _get_statement(self, codestring):
        """ Get code string as a statement - i.e. ending with a semicolon. """
        return codestring if codestring.endswith(';') else codestring + ';'

    def _get_comment(self, text):
        return '/* {} */'.format(text)

    def _declare_number_const(self, name, value):
        type_ = self.type_aliases[real]
        var = Variable(name, type=type_, value=value.evalf(type_.decimal_dig), attrs={value_const})
        decl = Declaration(var)
        return self._get_statement(self._print(decl))

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    @_as_macro_if_defined
    def _print_Mul(self, expr, **kwargs):
        return super()._print_Mul(expr, **kwargs)

    @_as_macro_if_defined
    def _print_Pow(self, expr):
        if 'Pow' in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if equal_valued(expr.exp, -1):
            literal_suffix = self._get_literal_suffix(real)
            return '1.0%s/%s' % (literal_suffix, self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One / 3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base), self._print(expr.exp))

    def _print_Mod(self, expr):
        num, den = expr.args
        if num.is_integer and den.is_integer:
            PREC = precedence(expr)
            snum, sden = [self.parenthesize(arg, PREC) for arg in expr.args]
            if num.is_nonnegative and den.is_nonnegative or (num.is_nonpositive and den.is_nonpositive):
                return f'{snum} % {sden}'
            return f'(({snum} % {sden}) + {sden}) % {sden}'
        return self._print_math_func(expr, known='fmod')

    def _print_Rational(self, expr):
        p, q = (int(expr.p), int(expr.q))
        suffix = self._get_literal_suffix(real)
        return '%d.0%s/%d.0%s' % (p, suffix, q, suffix)

    def _print_Indexed(self, expr):
        offset = getattr(expr.base, 'offset', S.Zero)
        strides = getattr(expr.base, 'strides', None)
        indices = expr.indices
        if strides is None or isinstance(strides, str):
            dims = expr.shape
            shift = S.One
            temp = ()
            if strides == 'C' or strides is None:
                traversal = reversed(range(expr.rank))
                indices = indices[::-1]
            elif strides == 'F':
                traversal = range(expr.rank)
            for i in traversal:
                temp += (shift,)
                shift *= dims[i]
            strides = temp
        flat_index = sum([x[0] * x[1] for x in zip(indices, strides)]) + offset
        return '%s[%s]' % (self._print(expr.base.label), self._print(flat_index))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    @_as_macro_if_defined
    def _print_NumberSymbol(self, expr):
        return super()._print_NumberSymbol(expr)

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

    def _print_Piecewise(self, expr):
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

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise, deep=False))

    def _print_MatrixElement(self, expr):
        return '{}[{}]'.format(self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True), expr.j + expr.i * expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._settings['dereference']:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError('Only iterable currently supported is Range')
        body = self._print(expr.body)
        return 'for ({target} = {start}; {target} < {stop}; {target} += {step}) {{\n{body}\n}}'.format(target=target, start=start, stop=stop, step=step, body=body)

    def _print_sign(self, func):
        return '((({0}) > 0) - (({0}) < 0))'.format(self._print(func.args[0]))

    def _print_Max(self, expr):
        if 'Max' in self.known_functions:
            return self._print_Function(expr)

        def inner_print_max(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return '((%(a)s > %(b)s) ? %(a)s : %(b)s)' % {'a': inner_print_max(args[:half]), 'b': inner_print_max(args[half:])}
        return inner_print_max(expr.args)

    def _print_Min(self, expr):
        if 'Min' in self.known_functions:
            return self._print_Function(expr)

        def inner_print_min(args):
            if len(args) == 1:
                return self._print(args[0])
            half = len(args) // 2
            return '((%(a)s < %(b)s) ? %(a)s : %(b)s)' % {'a': inner_print_min(args[:half]), 'b': inner_print_min(args[half:])}
        return inner_print_min(expr.args)

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

    def _get_func_suffix(self, type_):
        return self.type_func_suffixes[self.type_aliases.get(type_, type_)]

    def _get_literal_suffix(self, type_):
        return self.type_literal_suffixes[self.type_aliases.get(type_, type_)]

    def _get_math_macro_suffix(self, type_):
        alias = self.type_aliases.get(type_, type_)
        dflt = self.type_math_macro_suffixes.get(alias, '')
        return self.type_math_macro_suffixes.get(type_, dflt)

    def _print_Tuple(self, expr):
        return '{' + ', '.join((self._print(e) for e in expr)) + '}'
    _print_List = _print_Tuple

    def _print_Type(self, type_):
        self.headers.update(self.type_headers.get(type_, set()))
        self.macros.update(self.type_macros.get(type_, set()))
        return self._print(self.type_mappings.get(type_, type_.name))

    def _print_Declaration(self, decl):
        from sympy.codegen.cnodes import restrict
        var = decl.variable
        val = var.value
        if var.type == untyped:
            raise ValueError('C does not support untyped variables')
        if isinstance(var, Pointer):
            result = '{vc}{t} *{pc} {r}{s}'.format(vc='const ' if value_const in var.attrs else '', t=self._print(var.type), pc=' const' if pointer_const in var.attrs else '', r='restrict ' if restrict in var.attrs else '', s=self._print(var.symbol))
        elif isinstance(var, Variable):
            result = '{vc}{t} {s}'.format(vc='const ' if value_const in var.attrs else '', t=self._print(var.type), s=self._print(var.symbol))
        else:
            raise NotImplementedError('Unknown type of var: %s' % type(var))
        if val != None:
            result += ' = %s' % self._print(val)
        return result

    def _print_Float(self, flt):
        type_ = self.type_aliases.get(real, real)
        self.macros.update(self.type_macros.get(type_, set()))
        suffix = self._get_literal_suffix(type_)
        num = str(flt.evalf(type_.decimal_dig))
        if 'e' not in num and '.' not in num:
            num += '.0'
        num_parts = num.split('e')
        num_parts[0] = num_parts[0].rstrip('0')
        if num_parts[0].endswith('.'):
            num_parts[0] += '0'
        return 'e'.join(num_parts) + suffix

    @requires(headers={'stdbool.h'})
    def _print_BooleanTrue(self, expr):
        return 'true'

    @requires(headers={'stdbool.h'})
    def _print_BooleanFalse(self, expr):
        return 'false'

    def _print_Element(self, elem):
        if elem.strides == None:
            if elem.offset != None:
                raise ValueError('Expected strides when offset is given')
            idxs = ']['.join((self._print(arg) for arg in elem.indices))
        else:
            global_idx = sum([i * s for i, s in zip(elem.indices, elem.strides)])
            if elem.offset != None:
                global_idx += elem.offset
            idxs = self._print(global_idx)
        return '{symb}[{idxs}]'.format(symb=self._print(elem.symbol), idxs=idxs)

    def _print_CodeBlock(self, expr):
        """ Elements of code blocks printed as statements. """
        return '\n'.join([self._get_statement(self._print(i)) for i in expr.args])

    def _print_While(self, expr):
        return 'while ({condition}) {{\n{body}\n}}'.format(**expr.kwargs(apply=lambda arg: self._print(arg)))

    def _print_Scope(self, expr):
        return '{\n%s\n}' % self._print_CodeBlock(expr.body)

    @requires(headers={'stdio.h'})
    def _print_Print(self, expr):
        return 'printf({fmt}, {pargs})'.format(fmt=self._print(expr.format_string), pargs=', '.join((self._print(arg) for arg in expr.print_args)))

    def _print_FunctionPrototype(self, expr):
        pars = ', '.join((self._print(Declaration(arg)) for arg in expr.parameters))
        return '%s %s(%s)' % (tuple((self._print(arg) for arg in (expr.return_type, expr.name))) + (pars,))

    def _print_FunctionDefinition(self, expr):
        return '%s%s' % (self._print_FunctionPrototype(expr), self._print_Scope(expr))

    def _print_Return(self, expr):
        arg, = expr.args
        return 'return %s' % self._print(arg)

    def _print_CommaOperator(self, expr):
        return '(%s)' % ', '.join((self._print(arg) for arg in expr.args))

    def _print_Label(self, expr):
        if expr.body == none:
            return '%s:' % str(expr.name)
        if len(expr.body.args) == 1:
            return '%s:\n%s' % (str(expr.name), self._print_CodeBlock(expr.body))
        return '%s:\n{\n%s\n}' % (str(expr.name), self._print_CodeBlock(expr.body))

    def _print_goto(self, expr):
        return 'goto %s' % expr.label.name

    def _print_PreIncrement(self, expr):
        arg, = expr.args
        return '++(%s)' % self._print(arg)

    def _print_PostIncrement(self, expr):
        arg, = expr.args
        return '(%s)++' % self._print(arg)

    def _print_PreDecrement(self, expr):
        arg, = expr.args
        return '--(%s)' % self._print(arg)

    def _print_PostDecrement(self, expr):
        arg, = expr.args
        return '(%s)--' % self._print(arg)

    def _print_struct(self, expr):
        return '%(keyword)s %(name)s {\n%(lines)s}' % {'keyword': expr.__class__.__name__, 'name': expr.name, 'lines': ';\n'.join([self._print(decl) for decl in expr.declarations] + [''])}

    def _print_BreakToken(self, _):
        return 'break'

    def _print_ContinueToken(self, _):
        return 'continue'
    _print_union = _print_struct