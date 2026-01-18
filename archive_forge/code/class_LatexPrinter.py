from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
class LatexPrinter(Printer):
    printmethod = '_latex'
    _default_settings: dict[str, Any] = {'full_prec': False, 'fold_frac_powers': False, 'fold_func_brackets': False, 'fold_short_frac': None, 'inv_trig_style': 'abbreviated', 'itex': False, 'ln_notation': False, 'long_frac_ratio': None, 'mat_delim': '[', 'mat_str': None, 'mode': 'plain', 'mul_symbol': None, 'order': None, 'symbol_names': {}, 'root_notation': True, 'mat_symbol_style': 'plain', 'imaginary_unit': 'i', 'gothic_re_im': False, 'decimal_separator': 'period', 'perm_cyclic': True, 'parenthesize_super': True, 'min': None, 'max': None, 'diff_operator': 'd'}

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation', 'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', 'equation' or 'equation*'")
        if self._settings['fold_short_frac'] is None and self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True
        mul_symbol_table = {None: ' ', 'ldot': ' \\,.\\, ', 'dot': ' \\cdot ', 'times': ' \\times '}
        try:
            self._settings['mul_symbol_latex'] = mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if self._settings['mul_symbol'].strip() in ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']:
                self._settings['mul_symbol_latex_numbers'] = mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = self._settings['mul_symbol']
        self._delim_dict = {'(': ')', '[': ']'}
        imaginary_unit_table = {None: 'i', 'i': 'i', 'ri': '\\mathrm{i}', 'ti': '\\text{i}', 'j': 'j', 'rj': '\\mathrm{j}', 'tj': '\\text{j}'}
        imag_unit = self._settings['imaginary_unit']
        self._settings['imaginary_unit_latex'] = imaginary_unit_table.get(imag_unit, imag_unit)
        diff_operator_table = {None: 'd', 'd': 'd', 'rd': '\\mathrm{d}', 'td': '\\text{d}'}
        diff_operator = self._settings['diff_operator']
        self._settings['diff_operator_latex'] = diff_operator_table.get(diff_operator, diff_operator)

    def _add_parens(self, s) -> str:
        return '\\left({}\\right)'.format(s)

    def _add_parens_lspace(self, s) -> str:
        return '\\left( {}\\right)'.format(s)

    def parenthesize(self, item, level, is_neg=False, strict=False) -> str:
        prec_val = precedence_traditional(item)
        if is_neg and strict:
            return self._add_parens(self._print(item))
        if prec_val < level or (not strict and prec_val <= level):
            return self._add_parens(self._print(item))
        else:
            return self._print(item)

    def parenthesize_super(self, s):
        """
        Protect superscripts in s

        If the parenthesize_super option is set, protect with parentheses, else
        wrap in braces.
        """
        if '^' in s:
            if self._settings['parenthesize_super']:
                return self._add_parens(s)
            else:
                return '{{{}}}'.format(s)
        return s

    def doprint(self, expr) -> str:
        tex = Printer.doprint(self, expr)
        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return '$%s$' % tex
        elif self._settings['itex']:
            return '$$%s$$' % tex
        else:
            env_str = self._settings['mode']
            return '\\begin{%s}%s\\end{%s}' % (env_str, tex, env_str)

    def _needs_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not (expr.is_Integer and expr.is_nonnegative or (expr.is_Atom and (expr is not S.NegativeOne and expr.is_Rational is False)))

    def _needs_function_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        elif expr.is_Mul and (not self._mul_is_clean(expr)):
            return True
        elif expr.is_Pow and (not self._pow_is_clean(expr)):
            return True
        elif expr.is_Add or expr.is_Function:
            return True
        else:
            return False

    def _needs_mul_brackets(self, expr, first=False, last=False) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        if expr.is_Mul:
            if not first and expr.could_extract_minus_sign():
                return True
        elif precedence_traditional(expr) < PRECEDENCE['Mul']:
            return True
        elif expr.is_Relational:
            return True
        if expr.is_Piecewise:
            return True
        if any((expr.has(x) for x in (Mod,))):
            return True
        if not last and any((expr.has(x) for x in (Integral, Product, Sum))):
            return True
        return False

    def _needs_add_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        if expr.is_Relational:
            return True
        if any((expr.has(x) for x in (Mod,))):
            return True
        if expr.is_Add:
            return True
        return False

    def _mul_is_clean(self, expr) -> bool:
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr) -> bool:
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr: str, exp):
        if exp is not None:
            return '\\left(%s\\right)^{%s}' % (expr, exp)
        else:
            return expr

    def _print_Basic(self, expr):
        name = self._deal_with_super_sub(expr.__class__.__name__)
        if expr.args:
            ls = [self._print(o) for o in expr.args]
            s = '\\operatorname{{{}}}\\left({}\\right)'
            return s.format(name, ', '.join(ls))
        else:
            return '\\text{{{}}}'.format(name)

    def _print_bool(self, e: bool | BooleanTrue | BooleanFalse):
        return '\\text{%s}' % e
    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return '\\text{%s}' % e

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)
        tex = ''
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif term.could_extract_minus_sign():
                tex += ' - '
                term = -term
            else:
                tex += ' + '
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = '\\left(%s\\right)' % term_tex
            tex += term_tex
        return tex

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return '\\left( \\right)'
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', '\\;')
        term_tex = term_tex.replace('[', '\\left( ')
        term_tex = term_tex.replace(']', '\\right)')
        return term_tex

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation
        from sympy.utilities.exceptions import sympy_deprecation_warning
        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(f'\n                Setting Permutation.print_cyclic is deprecated. Instead use\n                init_printing(perm_cyclic={perm_cyclic}).\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-permutation-print_cyclic', stacklevel=8)
        else:
            perm_cyclic = self._settings.get('perm_cyclic', True)
        if perm_cyclic:
            return self._print_Cycle(expr)
        if expr.size == 0:
            return '\\left( \\right)'
        lower = [self._print(arg) for arg in expr.array_form]
        upper = [self._print(arg) for arg in range(len(lower))]
        row1 = ' & '.join(upper)
        row2 = ' & '.join(lower)
        mat = ' \\\\ '.join((row1, row2))
        return '\\begin{pmatrix} %s \\end{pmatrix}' % mat

    def _print_AppliedPermutation(self, expr):
        perm, var = expr.args
        return '\\sigma_{%s}(%s)' % (self._print(perm), self._print(var))

    def _print_Float(self, expr):
        dps = prec_to_dps(expr._prec)
        strip = False if self._settings['full_prec'] else True
        low = self._settings['min'] if 'min' in self._settings else None
        high = self._settings['max'] if 'max' in self._settings else None
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        separator = self._settings['mul_symbol_latex_numbers']
        if 'e' in str_real:
            mant, exp = str_real.split('e')
            if exp[0] == '+':
                exp = exp[1:]
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.', '{,}')
            return '%s%s10^{%s}' % (mant, separator, exp)
        elif str_real == '+inf':
            return '\\infty'
        elif str_real == '-inf':
            return '- \\infty'
        else:
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.', '{,}')
            return str_real

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return '%s \\times %s' % (self.parenthesize(vec1, PRECEDENCE['Mul']), self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return '\\nabla\\times %s' % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return '\\nabla\\cdot %s' % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return '%s \\cdot %s' % (self.parenthesize(vec1, PRECEDENCE['Mul']), self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return '\\nabla %s' % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        func = expr._expr
        return '\\Delta %s' % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Mul(self, expr: Expr):
        from sympy.simplify import fraction
        separator: str = self._settings['mul_symbol_latex']
        numbersep: str = self._settings['mul_symbol_latex_numbers']

        def convert(expr) -> str:
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)
                units, nonunits = sift(args, lambda x: (hasattr(x, '_scale_factor') or hasattr(x, 'is_physical_constant')) or (isinstance(x, Pow) and hasattr(x.base, 'is_physical_constant')), binary=True)
                prefixes, units = sift(units, lambda x: hasattr(x, '_scale_factor'), binary=True)
                return convert_args(nonunits + prefixes + units)

        def convert_args(args) -> str:
            _tex = last_term_tex = ''
            for i, term in enumerate(args):
                term_tex = self._print(term)
                if not (hasattr(term, '_scale_factor') or hasattr(term, 'is_physical_constant')):
                    if self._needs_mul_brackets(term, first=i == 0, last=i == len(args) - 1):
                        term_tex = '\\left(%s\\right)' % term_tex
                    if _between_two_numbers_p[0].search(last_term_tex) and _between_two_numbers_p[1].match(str(term)):
                        _tex += numbersep
                    elif _tex:
                        _tex += separator
                elif _tex:
                    _tex += separator
                _tex += term_tex
                last_term_tex = term_tex
            return _tex
        if isinstance(expr, Mul):
            args = expr.args
            if args[0] is S.One or any((isinstance(arg, Number) for arg in args[1:])):
                return convert_args(args)
        include_parens = False
        if expr.could_extract_minus_sign():
            expr = -expr
            tex = '- '
            if expr.is_Add:
                tex += '('
                include_parens = True
        else:
            tex = ''
        numer, denom = fraction(expr, exact=True)
        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            tex += convert(expr)
        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] and ldenom <= 2 and ('^' not in sdenom):
                if self._needs_mul_brackets(numer, last=False):
                    tex += '\\left(%s\\right) / %s' % (snumer, sdenom)
                else:
                    tex += '%s / %s' % (snumer, sdenom)
            elif ratio is not None and len(snumer.split()) > ratio * ldenom:
                if self._needs_mul_brackets(numer, last=True):
                    tex += '\\frac{1}{%s}%s\\left(%s\\right)' % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or len(convert(a * x).split()) > ratio * ldenom or b.is_commutative is x.is_commutative is False:
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += '\\frac{%s}{%s}%s\\left(%s\\right)' % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += '\\frac{%s}{%s}%s%s' % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += '\\frac{1}{%s}%s%s' % (sdenom, separator, snumer)
            else:
                tex += '\\frac{%s}{%s}' % (snumer, sdenom)
        if include_parens:
            tex += ')'
        return tex

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_PrimeIdeal(self, expr):
        p = self._print(expr.p)
        if expr.is_inert:
            return f'\\left({p}\\right)'
        alpha = self._print(expr.alpha.as_expr())
        return f'\\left({p}, {alpha}\\right)'

    def _print_Pow(self, expr: Pow):
        if expr.exp.is_Rational:
            p: int = expr.exp.p
            q: int = expr.exp.q
            if abs(p) == 1 and q != 1 and self._settings['root_notation']:
                base = self._print(expr.base)
                if q == 2:
                    tex = '\\sqrt{%s}' % base
                elif self._settings['itex']:
                    tex = '\\root{%d}{%s}' % (q, base)
                else:
                    tex = '\\sqrt[%d]{%s}' % (q, base)
                if expr.exp.is_negative:
                    return '\\frac{1}{%s}' % tex
                else:
                    return tex
            elif self._settings['fold_frac_powers'] and q != 1:
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                if expr.base.is_Symbol:
                    base = self.parenthesize_super(base)
                if expr.base.is_Function:
                    return self._print(expr.base, exp='%s/%s' % (p, q))
                return '%s^{%s/%s}' % (base, p, q)
            elif expr.exp.is_negative and expr.base.is_commutative:
                if expr.base == 1:
                    return '%s^{%s}' % (expr.base, expr.exp)
                if expr.base.is_Rational:
                    base_p: int = expr.base.p
                    base_q: int = expr.base.q
                    if base_p * base_q == abs(base_q):
                        if expr.exp == -1:
                            return '\\frac{1}{\\frac{%s}{%s}}' % (base_p, base_q)
                        else:
                            return '\\frac{1}{(\\frac{%s}{%s})^{%s}}' % (base_p, base_q, abs(expr.exp))
                return self._print_Mul(expr)
        if expr.base.is_Function:
            return self._print(expr.base, exp=self._print(expr.exp))
        tex = '%s^{%s}'
        return self._helper_print_standard_power(expr, tex)

    def _helper_print_standard_power(self, expr, template: str) -> str:
        exp = self._print(expr.exp)
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        if expr.base.is_Symbol:
            base = self.parenthesize_super(base)
        elif isinstance(expr.base, Derivative) and base.startswith('\\left(') and re.match('\\\\left\\(\\\\d?d?dot', base) and base.endswith('\\right)'):
            base = base[6:-7]
        return template % (base, exp)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = '\\sum_{%s=%s}^{%s} ' % tuple([self._print(i) for i in expr.limits[0]])
        else:

            def _format_ineq(l):
                return '%s \\leq %s \\leq %s' % tuple([self._print(s) for s in (l[1], l[0], l[2])])
            tex = '\\sum_{\\substack{%s}} ' % str.join('\\\\', [_format_ineq(l) for l in expr.limits])
        if isinstance(expr.function, Add):
            tex += '\\left(%s\\right)' % self._print(expr.function)
        else:
            tex += self._print(expr.function)
        return tex

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = '\\prod_{%s=%s}^{%s} ' % tuple([self._print(i) for i in expr.limits[0]])
        else:

            def _format_ineq(l):
                return '%s \\leq %s \\leq %s' % tuple([self._print(s) for s in (l[1], l[0], l[2])])
            tex = '\\prod_{\\substack{%s}} ' % str.join('\\\\', [_format_ineq(l) for l in expr.limits])
        if isinstance(expr.function, Add):
            tex += '\\left(%s\\right)' % self._print(expr.function)
        else:
            tex += self._print(expr.function)
        return tex

    def _print_BasisDependent(self, expr: 'BasisDependent'):
        from sympy.vector import Vector
        o1: list[str] = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = '\\left(' + self._print(v) + '\\right)'
                    o1.append(' + ' + arg_str + k._latex_form)
        outstr = ''.join(o1)
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{' + tex_base + '}' + '_{%s}' % ','.join(map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Idx(self, expr):
        label = self._print(expr.label)
        if expr.upper is not None:
            upper = self._print(expr.upper)
            if expr.lower is not None:
                lower = self._print(expr.lower)
            else:
                lower = self._print(S.Zero)
            interval = '{lower}\\mathrel{{..}}\\nobreak {upper}'.format(lower=lower, upper=upper)
            return '{{{label}}}_{{{interval}}}'.format(label=label, interval=interval)
        return label

    def _print_Derivative(self, expr):
        if requires_partial(expr.expr):
            diff_symbol = '\\partial'
        else:
            diff_symbol = self._settings['diff_operator_latex']
        tex = ''
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += '%s %s' % (diff_symbol, self._print(x))
            else:
                tex += '%s %s^{%s}' % (diff_symbol, self.parenthesize_super(self._print(x)), self._print(num))
        if dim == 1:
            tex = '\\frac{%s}{%s}' % (diff_symbol, tex)
        else:
            tex = '\\frac{%s^{%s}}{%s}' % (diff_symbol, self._print(dim), tex)
        if any((i.could_extract_minus_sign() for i in expr.args)):
            return '%s %s' % (tex, self.parenthesize(expr.expr, PRECEDENCE['Mul'], is_neg=True, strict=True))
        return '%s %s' % (tex, self.parenthesize(expr.expr, PRECEDENCE['Mul'], is_neg=False, strict=True))

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = '\\\\ '.join((e[0] + '=' + e[1] for e in zip(latex_old, latex_new)))
        return '\\left. %s \\right|_{\\substack{ %s }}' % (latex_expr, latex_subs)

    def _print_Integral(self, expr):
        tex, symbols = ('', [])
        diff_symbol = self._settings['diff_operator_latex']
        if len(expr.limits) <= 4 and all((len(lim) == 1 for lim in expr.limits)):
            tex = '\\i' + 'i' * (len(expr.limits) - 1) + 'nt'
            symbols = ['\\, %s%s' % (diff_symbol, self._print(symbol[0])) for symbol in expr.limits]
        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += '\\int'
                if len(lim) > 1:
                    if self._settings['mode'] != 'inline' and (not self._settings['itex']):
                        tex += '\\limits'
                    if len(lim) == 3:
                        tex += '_{%s}^{%s}' % (self._print(lim[1]), self._print(lim[2]))
                    if len(lim) == 2:
                        tex += '^{%s}' % self._print(lim[1])
                symbols.insert(0, '\\, %s%s' % (diff_symbol, self._print(symbol)))
        return '%s %s%s' % (tex, self.parenthesize(expr.function, PRECEDENCE['Mul'], is_neg=any((i.could_extract_minus_sign() for i in expr.args)), strict=True), ''.join(symbols))

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args
        tex = '\\lim_{%s \\to ' % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += '%s}' % self._print(z0)
        else:
            tex += '%s^%s}' % (self._print(z0), self._print(dir))
        if isinstance(e, AssocOp):
            return '%s\\left(%s\\right)' % (tex, self._print(e))
        else:
            return '%s %s' % (tex, self._print(e))

    def _hprint_Function(self, func: str) -> str:
        """
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \\operatorname{} around it and be
            mindful of undercores in the name
        """
        func = self._deal_with_super_sub(func)
        superscriptidx = func.find('^')
        subscriptidx = func.find('_')
        if func in accepted_latex_functions:
            name = '\\%s' % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or (superscriptidx == 1):
            name = func
        elif superscriptidx > 0 and subscriptidx > 0:
            name = '\\operatorname{%s}%s' % (func[:min(subscriptidx, superscriptidx)], func[min(subscriptidx, superscriptidx):])
        elif superscriptidx > 0:
            name = '\\operatorname{%s}%s' % (func[:superscriptidx], func[superscriptidx:])
        elif subscriptidx > 0:
            name = '\\operatorname{%s}%s' % (func[:subscriptidx], func[subscriptidx:])
        else:
            name = '\\operatorname{%s}' % func
        return name

    def _print_Function(self, expr: Function, exp=None) -> str:
        """
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\\sin, \\cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \\operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        """
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and (not isinstance(expr, AppliedUndef)):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [str(self._print(arg)) for arg in expr.args]
            inv_trig_style = self._settings['inv_trig_style']
            inv_trig_power_case = False
            can_fold_brackets = self._settings['fold_func_brackets'] and len(args) == 1 and (not self._needs_function_brackets(expr.args[0]))
            inv_trig_table = ['asin', 'acos', 'atan', 'acsc', 'asec', 'acot', 'asinh', 'acosh', 'atanh', 'acsch', 'asech', 'acoth']
            if func in inv_trig_table:
                if inv_trig_style == 'abbreviated':
                    pass
                elif inv_trig_style == 'full':
                    func = ('ar' if func[-1] == 'h' else 'arc') + func[1:]
                elif inv_trig_style == 'power':
                    func = func[1:]
                    inv_trig_power_case = True
                    if exp is not None:
                        can_fold_brackets = False
            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = '\\%s^{-1}' % func
                else:
                    name = '\\operatorname{%s}^{-1}' % func
            elif exp is not None:
                func_tex = self._hprint_Function(func)
                func_tex = self.parenthesize_super(func_tex)
                name = '%s^{%s}' % (func_tex, exp)
            else:
                name = self._hprint_Function(func)
            if can_fold_brackets:
                if func in accepted_latex_functions:
                    name += ' {%s}'
                else:
                    name += '%s'
            else:
                name += '{\\left(%s \\right)}'
            if inv_trig_power_case and exp is not None:
                name += '^{%s}' % exp
            return name % ','.join(args)

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return '{%s}_{\\circ}\\left({%s}\\right)' % (self._print(expr.function), self._print(expr.expr))

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: '\\delta', gamma: '\\Gamma', lowergamma: '\\gamma', beta: '\\operatorname{B}', DiracDelta: '\\delta', Chi: '\\operatorname{Chi}'}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args
        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))
        tex = '\\left( %s \\mapsto %s \\right)' % (symbols, self._print(expr))
        return tex

    def _print_IdentityFunction(self, expr):
        return '\\left( x \\mapsto x \\right)'

    def _hprint_variadic_function(self, expr, exp=None) -> str:
        args = sorted(expr.args, key=default_sort_key)
        texargs = ['%s' % self._print(symbol) for symbol in args]
        tex = '\\%s\\left(%s\\right)' % (str(expr.func).lower(), ', '.join(texargs))
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex
    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = '\\left\\lfloor{%s}\\right\\rfloor' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = '\\left\\lceil{%s}\\right\\rceil' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings['ln_notation']:
            tex = '\\log{\\left(%s \\right)}' % self._print(expr.args[0])
        else:
            tex = '\\ln{\\left(%s \\right)}' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = '\\left|{%s}\\right|' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_re(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = '\\Re{%s}' % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = '\\operatorname{{re}}{{{}}}'.format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))
        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = '\\Im{%s}' % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = '\\operatorname{{im}}{{{}}}'.format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))
        return self._do_exponent(tex, exp)

    def _print_Not(self, e):
        from sympy.logic.boolalg import Equivalent, Implies
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], '\\not\\Leftrightarrow')
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], '\\not\\Rightarrow')
        if e.args[0].is_Boolean:
            return '\\neg \\left(%s\\right)' % self._print(e.args[0])
        else:
            return '\\neg %s' % self._print(e.args[0])

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and (not arg.is_Not):
            tex = '\\left(%s\\right)' % self._print(arg)
        else:
            tex = '%s' % self._print(arg)
        for arg in args[1:]:
            if arg.is_Boolean and (not arg.is_Not):
                tex += ' %s \\left(%s\\right)' % (char, self._print(arg))
            else:
                tex += ' %s %s' % (char, self._print(arg))
        return tex

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, '\\wedge')

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, '\\vee')

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, '\\veebar')

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or '\\Rightarrow')

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or '\\Leftrightarrow')

    def _print_conjugate(self, expr, exp=None):
        tex = '\\overline{%s}' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = '\\operatorname{polar\\_lift}'
        arg = '{\\left(%s \\right)}' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}%s' % (func, exp, arg)
        else:
            return '%s%s' % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        tex = 'e^{%s}' % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_Exp1(self, expr, exp=None):
        return 'e'

    def _print_elliptic_k(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return 'K^{%s}%s' % (exp, tex)
        else:
            return 'K%s' % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = '\\left(%s\\middle| %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return 'F^{%s}%s' % (exp, tex)
        else:
            return 'F%s' % tex

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = '\\left(%s\\middle| %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return 'E^{%s}%s' % (exp, tex)
        else:
            return 'E%s' % tex

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = '\\left(%s; %s\\middle| %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]), self._print(expr.args[2]))
        else:
            tex = '\\left(%s\\middle| %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return '\\Pi^{%s}%s' % (exp, tex)
        else:
            return '\\Pi%s' % tex

    def _print_beta(self, expr, exp=None):
        x = expr.args[0]
        y = expr.args[0] if len(expr.args) == 1 else expr.args[1]
        tex = f'\\left({x}, {y}\\right)'
        if exp is not None:
            return '\\operatorname{B}^{%s}%s' % (exp, tex)
        else:
            return '\\operatorname{B}%s' % tex

    def _print_betainc(self, expr, exp=None, operator='B'):
        largs = [self._print(arg) for arg in expr.args]
        tex = '\\left(%s, %s\\right)' % (largs[0], largs[1])
        if exp is not None:
            return '\\operatorname{%s}_{(%s, %s)}^{%s}%s' % (operator, largs[2], largs[3], exp, tex)
        else:
            return '\\operatorname{%s}_{(%s, %s)}%s' % (operator, largs[2], largs[3], tex)

    def _print_betainc_regularized(self, expr, exp=None):
        return self._print_betainc(expr, exp, operator='I')

    def _print_uppergamma(self, expr, exp=None):
        tex = '\\left(%s, %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return '\\Gamma^{%s}%s' % (exp, tex)
        else:
            return '\\Gamma%s' % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = '\\left(%s, %s\\right)' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return '\\gamma^{%s}%s' % (exp, tex)
        else:
            return '\\gamma%s' % tex

    def _hprint_one_arg_func(self, expr, exp=None) -> str:
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}%s' % (self._print(expr.func), exp, tex)
        else:
            return '%s%s' % (self._print(expr.func), tex)
    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '\\operatorname{Chi}^{%s}%s' % (exp, tex)
        else:
            return '\\operatorname{Chi}%s' % tex

    def _print_expint(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[1])
        nu = self._print(expr.args[0])
        if exp is not None:
            return '\\operatorname{E}_{%s}^{%s}%s' % (nu, exp, tex)
        else:
            return '\\operatorname{E}_{%s}%s' % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return 'S^{%s}%s' % (exp, tex)
        else:
            return 'S%s' % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return 'C^{%s}%s' % (exp, tex)
        else:
            return 'C%s' % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = '!%s' % self.parenthesize(expr.args[0], PRECEDENCE['Func'])
        if exp is not None:
            return '\\left(%s\\right)^{%s}' % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = '%s!' % self.parenthesize(expr.args[0], PRECEDENCE['Func'])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_factorial2(self, expr, exp=None):
        tex = '%s!!' % self.parenthesize(expr.args[0], PRECEDENCE['Func'])
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = '{\\binom{%s}{%s}}' % (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return '%s^{%s}' % (tex, exp)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = '%s' % self.parenthesize(n, PRECEDENCE['Func'])
        tex = '{%s}^{\\left(%s\\right)}' % (base, self._print(k))
        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = '%s' % self.parenthesize(k, PRECEDENCE['Func'])
        tex = '{\\left(%s\\right)}_{%s}' % (self._print(n), sub)
        return self._do_exponent(tex, exp)

    def _hprint_BesselBase(self, expr, exp, sym: str) -> str:
        tex = '%s' % sym
        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = '%s^{%s}' % (tex, exp)
            else:
                need_exp = True
        tex = '%s_{%s}\\left(%s\\right)' % (tex, self._print(expr.order), self._print(expr.argument))
        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex

    def _hprint_vec(self, vec) -> str:
        if not vec:
            return ''
        s = ''
        for i in vec[:-1]:
            s += '%s, ' % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation='') -> str:
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '%s^{%s}%s' % (notation, exp, tex)
        else:
            return '%s%s' % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation='') -> str:
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '{%s^\\prime}^{%s}%s' % (notation, exp, tex)
        else:
            return '%s^\\prime%s' % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')

    def _print_hyper(self, expr, exp=None):
        tex = '{{}_{%s}F_{%s}\\left(\\begin{matrix} %s \\\\ %s \\end{matrix}\\middle| {%s} \\right)}' % (self._print(len(expr.ap)), self._print(len(expr.bq)), self._hprint_vec(expr.ap), self._hprint_vec(expr.bq), self._print(expr.argument))
        if exp is not None:
            tex = '{%s}^{%s}' % (tex, exp)
        return tex

    def _print_meijerg(self, expr, exp=None):
        tex = '{G_{%s, %s}^{%s, %s}\\left(\\begin{matrix} %s & %s \\\\%s & %s \\end{matrix} \\middle| {%s} \\right)}' % (self._print(len(expr.ap)), self._print(len(expr.bq)), self._print(len(expr.bm)), self._print(len(expr.an)), self._hprint_vec(expr.an), self._hprint_vec(expr.aother), self._hprint_vec(expr.bm), self._hprint_vec(expr.bother), self._print(expr.argument))
        if exp is not None:
            tex = '{%s}^{%s}' % (tex, exp)
        return tex

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '\\eta^{%s}%s' % (exp, tex)
        return '\\eta%s' % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = '\\left(%s, %s\\right)' % tuple(map(self._print, expr.args))
        else:
            tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '\\zeta^{%s}%s' % (exp, tex)
        return '\\zeta%s' % tex

    def _print_stieltjes(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = '_{%s}\\left(%s\\right)' % tuple(map(self._print, expr.args))
        else:
            tex = '_{%s}' % self._print(expr.args[0])
        if exp is not None:
            return '\\gamma%s^{%s}' % (tex, exp)
        return '\\gamma%s' % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = '\\left(%s, %s, %s\\right)' % tuple(map(self._print, expr.args))
        if exp is None:
            return '\\Phi%s' % tex
        return '\\Phi^{%s}%s' % (exp, tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = '\\left(%s\\right)' % z
        if exp is None:
            return '\\operatorname{Li}_{%s}%s' % (s, tex)
        return '\\operatorname{Li}_{%s}^{%s}%s' % (s, exp, tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = 'P_{%s}^{\\left(%s,%s\\right)}\\left(%s\\right)' % (n, a, b, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = 'C_{%s}^{\\left(%s\\right)}\\left(%s\\right)' % (n, a, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = 'T_{%s}\\left(%s\\right)' % (n, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = 'U_{%s}\\left(%s\\right)' % (n, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = 'P_{%s}\\left(%s\\right)' % (n, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = 'P_{%s}^{\\left(%s\\right)}\\left(%s\\right)' % (n, a, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = 'H_{%s}\\left(%s\\right)' % (n, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = 'L_{%s}\\left(%s\\right)' % (n, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = 'L_{%s}^{\\left(%s\\right)}\\left(%s\\right)' % (n, a, x)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = 'Y_{%s}^{%s}\\left(%s,%s\\right)' % (n, m, theta, phi)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = 'Z_{%s}^{%s}\\left(%s,%s\\right)' % (n, m, theta, phi)
        if exp is not None:
            tex = '\\left(' + tex + '\\right)^{%s}' % exp
        return tex

    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        a, q, z = map(self._print, args)
        sup = '^{\\prime}' if prime else ''
        exp = '' if not exp else '^{%s}' % exp
        return '%s%s\\left(%s, %s, %s\\right)%s' % (character, sup, a, q, z, exp)

    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions('C', expr.args, exp=exp)

    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions('S', expr.args, exp=exp)

    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions('C', expr.args, prime=True, exp=exp)

    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions('S', expr.args, prime=True, exp=exp)

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ''
            p = expr.p
            if expr.p < 0:
                sign = '- '
                p = -p
            if self._settings['fold_short_frac']:
                return '%s%d / %d' % (sign, p, expr.q)
            return '%s\\frac{%d}{%d}' % (sign, p, expr.q)
        else:
            return self._print(expr.p)

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any((p != S.Zero for p in expr.point)) or len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif expr.variables:
                s += self._print(expr.variables[0])
            s += '\\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return 'O\\left(%s\\right)' % s

    def _print_Symbol(self, expr: Symbol, style='plain'):
        name: str = self._settings['symbol_names'].get(expr)
        if name is not None:
            return name
        return self._deal_with_super_sub(expr.name, style=style)
    _print_RandomSymbol = _print_Symbol

    def _deal_with_super_sub(self, string: str, style='plain') -> str:
        if '{' in string:
            name, supers, subs = (string, [], [])
        else:
            name, supers, subs = split_super_sub(string)
            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]
        if style == 'bold':
            name = '\\mathbf{{{}}}'.format(name)
        if supers:
            name += '^{%s}' % ' '.join(supers)
        if subs:
            name += '_{%s}' % ' '.join(subs)
        return name

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = '\\gt'
            lt = '\\lt'
        else:
            gt = '>'
            lt = '<'
        charmap = {'==': '=', '>': gt, '<': lt, '>=': '\\geq', '<=': '\\leq', '!=': '\\neq'}
        return '%s %s %s' % (self._print(expr.lhs), charmap[expr.rel_op], self._print(expr.rhs))

    def _print_Piecewise(self, expr):
        ecpairs = ['%s & \\text{for}\\: %s' % (self._print(e), self._print(c)) for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append('%s & \\text{otherwise}' % self._print(expr.args[-1].expr))
        else:
            ecpairs.append('%s & \\text{for}\\: %s' % (self._print(expr.args[-1].expr), self._print(expr.args[-1].cond)))
        tex = '\\begin{cases} %s \\end{cases}'
        return tex % ' \\\\'.join(ecpairs)

    def _print_matrix_contents(self, expr):
        lines = []
        for line in range(expr.rows):
            lines.append(' & '.join([self._print(i) for i in expr[line, :]]))
        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            elif (expr.cols <= 10) is True:
                mat_str = 'matrix'
            else:
                mat_str = 'array'
        out_str = '\\begin{%MATSTR%}%s\\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c' * expr.cols + '}%s')
        return out_str % '\\\\'.join(lines)

    def _print_MatrixBase(self, expr):
        out_str = self._print_matrix_contents(expr)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = '\\left' + left_delim + out_str + '\\right' + right_delim
        return out_str

    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '_{%s, %s}' % (self._print(expr.i), self._print(expr.j))

    def _print_MatrixSlice(self, expr):

        def latexslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = None
            if x[1] == dim:
                x[1] = None
            return ':'.join((self._print(xi) if xi is not None else '' for xi in x))
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '\\left[' + latexslice(expr.rowslice, expr.parent.rows) + ', ' + latexslice(expr.colslice, expr.parent.cols) + '\\right]'

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if not isinstance(mat, MatrixSymbol) and (not isinstance(mat, BlockMatrix)) and mat.is_MatrixExpr:
            return '\\left(%s\\right)^{T}' % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return '\\left(%s\\right)^{T}' % s
            else:
                return '%s^{T}' % s

    def _print_Trace(self, expr):
        mat = expr.arg
        return '\\operatorname{tr}\\left(%s \\right)' % self._print(mat)

    def _print_Adjoint(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if not isinstance(mat, MatrixSymbol) and (not isinstance(mat, BlockMatrix)) and mat.is_MatrixExpr:
            return '\\left(%s\\right)^{\\dagger}' % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return '\\left(%s\\right)^{\\dagger}' % s
            else:
                return '%s^{\\dagger}' % s

    def _print_MatMul(self, expr):
        from sympy import MatMul
        parens = lambda x: self._print(x) if isinstance(x, Mul) and (not isinstance(x, MatMul)) else self.parenthesize(x, precedence_traditional(expr), False)
        args = list(expr.args)
        if expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            return '- ' + ' '.join(map(parens, args))
        else:
            return ' '.join(map(parens, args))

    def _print_Determinant(self, expr):
        mat = expr.arg
        if mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return '\\left|{%s}\\right|' % self._print_matrix_contents(mat.blocks)
            return '\\left|{%s}\\right|' % self._print(mat)
        return '\\left|{%s}\\right|' % self._print_matrix_contents(mat)

    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return '\\left(%s \\bmod %s\\right)^{%s}' % (self.parenthesize(expr.args[0], PRECEDENCE['Mul'], strict=True), self.parenthesize(expr.args[1], PRECEDENCE['Mul'], strict=True), exp)
        return '%s \\bmod %s' % (self.parenthesize(expr.args[0], PRECEDENCE['Mul'], strict=True), self.parenthesize(expr.args[1], PRECEDENCE['Mul'], strict=True))

    def _print_HadamardProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize
        return ' \\circ '.join((parens(arg, prec, strict=True) for arg in args))

    def _print_HadamardPower(self, expr):
        if precedence_traditional(expr.exp) < PRECEDENCE['Mul']:
            template = '%s^{\\circ \\left({%s}\\right)}'
        else:
            template = '%s^{\\circ {%s}}'
        return self._helper_print_standard_power(expr, template)

    def _print_KroneckerProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize
        return ' \\otimes '.join((parens(arg, prec, strict=True) for arg in args))

    def _print_MatPow(self, expr):
        base, exp = (expr.base, expr.exp)
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol) and base.is_MatrixExpr:
            return '\\left(%s\\right)^{%s}' % (self._print(base), self._print(exp))
        else:
            base_str = self._print(base)
            if '^' in base_str:
                return '\\left(%s\\right)^{%s}' % (base_str, self._print(exp))
            else:
                return '%s^{%s}' % (base_str, self._print(exp))

    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings['mat_symbol_style'])

    def _print_ZeroMatrix(self, Z):
        return '0' if self._settings['mat_symbol_style'] == 'plain' else '\\mathbf{0}'

    def _print_OneMatrix(self, O):
        return '1' if self._settings['mat_symbol_style'] == 'plain' else '\\mathbf{1}'

    def _print_Identity(self, I):
        return '\\mathbb{I}' if self._settings['mat_symbol_style'] == 'plain' else '\\mathbf{I}'

    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        return 'P_{%s}' % perm_str

    def _print_NDimArray(self, expr: NDimArray):
        if expr.rank() == 0:
            return self._print(expr[()])
        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            elif expr.rank() == 0 or expr.shape[-1] <= 10:
                mat_str = 'matrix'
            else:
                mat_str = 'array'
        block_str = '\\begin{%MATSTR%}%s\\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            block_str = block_str.replace('%s', '{}%s')
        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = '\\left' + left_delim + block_str + '\\right' + right_delim
        if expr.rank() == 0:
            return block_str % ''
        level_str: list[list[str]] = [[] for i in range(expr.rank() + 1)]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank() - 1, -1, -1):
                if len(level_str[back_outer_i + 1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(' & '.join(level_str[back_outer_i + 1]))
                else:
                    level_str[back_outer_i].append(block_str % '\\\\'.join(level_str[back_outer_i + 1]))
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = '\\left[' + level_str[back_outer_i][-1] + '\\right]'
                even = not even
                level_str[back_outer_i + 1] = []
        out_str = level_str[0][0]
        if expr.rank() % 2 == 1:
            out_str = block_str % out_str
        return out_str

    def _printer_tensor_indices(self, name, indices, index_map: dict):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if (index in index_map or prev_map) and last_valence == new_valence:
                out_str += ','
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += '}'
                if index.is_up:
                    out_str += '{}^{'
                else:
                    out_str += '{}_{'
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += '='
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += '}'
        return out_str

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices, {})

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        sign, args = expr._get_args_for_traditional_printer()
        return sign + ''.join([self.parenthesize(arg, precedence(expr)) for arg in args])

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return '{}%s{%s}' % ('^' if expr.is_up else '_', self._print(expr.args[0]))

    def _print_PartialDerivative(self, expr):
        if len(expr.variables) == 1:
            return '\\frac{\\partial}{\\partial {%s}}{%s}' % (self._print(expr.variables[0]), self.parenthesize(expr.expr, PRECEDENCE['Mul'], False))
        else:
            return '\\frac{\\partial^{%s}}{%s}{%s}' % (len(expr.variables), ' '.join(['\\partial {%s}' % self._print(i) for i in expr.variables]), self.parenthesize(expr.expr, PRECEDENCE['Mul'], False))

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return '{{%s}_{%s}}' % (self.parenthesize(expr.name, PRECEDENCE['Func'], True), ', '.join([f'{self._print(i)}' for i in expr.indices]))

    def _print_UniversalSet(self, expr):
        return '\\mathbb{U}'

    def _print_frac(self, expr, exp=None):
        if exp is None:
            return '\\operatorname{frac}{\\left(%s\\right)}' % self._print(expr.args[0])
        else:
            return '\\operatorname{frac}{\\left(%s\\right)}^{%s}' % (self._print(expr.args[0]), exp)

    def _print_tuple(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            sep = ';'
        elif self._settings['decimal_separator'] == 'period':
            sep = ','
        else:
            raise ValueError('Unknown Decimal Separator')
        if len(expr) == 1:
            return self._add_parens_lspace(self._print(expr[0]) + sep)
        else:
            return self._add_parens_lspace((sep + ' \\  ').join([self._print(i) for i in expr]))

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return ' \\otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return ' \\wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            return '\\left[ %s\\right]' % '; \\  '.join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return '\\left[ %s\\right]' % ', \\  '.join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []
        for key in keys:
            val = d[key]
            items.append('%s : %s' % (self._print(key), self._print(val)))
        return '\\left\\{ %s\\right\\}' % ', \\  '.join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = '\\delta\\left(%s\\right)' % self._print(expr.args[0])
        else:
            tex = '\\delta^{\\left( %s \\right)}\\left( %s \\right)' % (self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = '\\left(%s\\right)^{%s}' % (tex, exp)
        return tex

    def _print_SingularityFunction(self, expr, exp=None):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = '{\\left\\langle %s \\right\\rangle}^{%s}' % (shift, power)
        if exp is not None:
            tex = '{\\left({\\langle %s \\rangle}^{%s}\\right)}^{%s}' % (shift, power, exp)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join((self._print(arg) for arg in expr.pargs))
        tex = '\\theta\\left(%s\\right)' % pargs
        if exp:
            tex = '\\left(%s\\right)^{%s}' % (tex, exp)
        return tex

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = '\\delta_{%s %s}' % (i, j)
        else:
            tex = '\\delta_{%s, %s}' % (i, j)
        if exp is not None:
            tex = '\\left(%s\\right)^{%s}' % (tex, exp)
        return tex

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all((x.is_Atom for x in expr.args)):
            tex = '\\varepsilon_{%s}' % ' '.join(indices)
        else:
            tex = '\\varepsilon_{%s}' % ', '.join(indices)
        if exp:
            tex = '\\left(%s\\right)^{%s}' % (tex, exp)
        return tex

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return '\\text{Domain: }' + self._print(d.symbols) + ' \\in ' + self._print(d.set)
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = '; '.join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ', '.join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return '\\left\\{%s\\right\\}' % items
    _print_frozenset = _print_set

    def _print_Range(self, s):

        def _print_symbolic_range():
            if s.args[0] == 0:
                if s.args[2] == 1:
                    cont = self._print(s.args[1])
                else:
                    cont = ', '.join((self._print(arg) for arg in s.args))
            elif s.args[2] == 1:
                cont = ', '.join((self._print(arg) for arg in s.args[:2]))
            else:
                cont = ', '.join((self._print(arg) for arg in s.args))
            return f'\\text{{Range}}\\left({cont}\\right)'
        dots = object()
        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = (dots, -1, 0, 1, dots)
            else:
                printset = (dots, 1, 0, -1, dots)
        elif s.start.is_infinite:
            printset = (dots, s[-1] - s.step, s[-1])
        elif s.stop.is_infinite:
            it = iter(s)
            printset = (next(it), next(it), dots)
        elif s.is_empty is not None:
            if (s.size < 4) == True:
                printset = tuple(s)
            elif s.is_iterable:
                it = iter(s)
                printset = (next(it), next(it), dots, s[-1])
            else:
                return _print_symbolic_range()
        else:
            return _print_symbolic_range()
        return '\\left\\{' + ', '.join((self._print(el) if el is not dots else '\\ldots' for el in printset)) + '\\right\\}'

    def __print_number_polynomial(self, expr, letter, exp=None):
        if len(expr.args) == 2:
            if exp is not None:
                return '%s_{%s}^{%s}\\left(%s\\right)' % (letter, self._print(expr.args[0]), exp, self._print(expr.args[1]))
            return '%s_{%s}\\left(%s\\right)' % (letter, self._print(expr.args[0]), self._print(expr.args[1]))
        tex = '%s_{%s}' % (letter, self._print(expr.args[0]))
        if exp is not None:
            tex = '%s^{%s}' % (tex, exp)
        return tex

    def _print_bernoulli(self, expr, exp=None):
        return self.__print_number_polynomial(expr, 'B', exp)

    def _print_genocchi(self, expr, exp=None):
        return self.__print_number_polynomial(expr, 'G', exp)

    def _print_bell(self, expr, exp=None):
        if len(expr.args) == 3:
            tex1 = 'B_{%s, %s}' % (self._print(expr.args[0]), self._print(expr.args[1]))
            tex2 = '\\left(%s\\right)' % ', '.join((self._print(el) for el in expr.args[2]))
            if exp is not None:
                tex = '%s^{%s}%s' % (tex1, exp, tex2)
            else:
                tex = tex1 + tex2
            return tex
        return self.__print_number_polynomial(expr, 'B', exp)

    def _print_fibonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, 'F', exp)

    def _print_lucas(self, expr, exp=None):
        tex = 'L_{%s}' % self._print(expr.args[0])
        if exp is not None:
            tex = '%s^{%s}' % (tex, exp)
        return tex

    def _print_tribonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, 'T', exp)

    def _print_SeqFormula(self, s):
        dots = object()
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            return '\\left\\{%s\\right\\}_{%s=%s}^{%s}' % (self._print(s.formula), self._print(s.variables[0]), self._print(s.start), self._print(s.stop))
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2), s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
        else:
            printset = tuple(s)
        return '\\left[' + ', '.join((self._print(el) if el is not dots else '\\ldots' for el in printset)) + '\\right]'
    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return '\\left\\{%s\\right\\}' % self._print(i.start)
        else:
            if i.left_open:
                left = '('
            else:
                left = '['
            if i.right_open:
                right = ')'
            else:
                right = ']'
            return '\\left%s%s, %s\\right%s' % (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return '\\left\\langle %s, %s\\right\\rangle' % (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return ' \\cup '.join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return ' \\setminus '.join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return ' \\cap '.join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return ' \\triangle '.join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and (not has_variety(p.sets)):
            return self.parenthesize(p.sets[0], prec) + '^{%d}' % len(p.sets)
        return ' \\times '.join((self.parenthesize(set, prec) for set in p.sets))

    def _print_EmptySet(self, e):
        return '\\emptyset'

    def _print_Naturals(self, n):
        return '\\mathbb{N}'

    def _print_Naturals0(self, n):
        return '\\mathbb{N}_0'

    def _print_Integers(self, i):
        return '\\mathbb{Z}'

    def _print_Rationals(self, i):
        return '\\mathbb{Q}'

    def _print_Reals(self, i):
        return '\\mathbb{R}'

    def _print_Complexes(self, i):
        return '\\mathbb{C}'

    def _print_ImageSet(self, s):
        expr = s.lamda.expr
        sig = s.lamda.signature
        xys = ((self._print(x), self._print(y)) for x, y in zip(sig, s.base_sets))
        xinys = ', '.join(('%s \\in %s' % xy for xy in xys))
        return '\\left\\{%s\\; \\middle|\\; %s\\right\\}' % (self._print(expr), xinys)

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return '\\left\\{%s\\; \\middle|\\; %s \\right\\}' % (vars_print, self._print(s.condition))
        return '\\left\\{%s\\; \\middle|\\; %s \\in %s \\wedge %s \\right\\}' % (vars_print, vars_print, self._print(s.base_set), self._print(s.condition))

    def _print_PowerSet(self, expr):
        arg_print = self._print(expr.args[0])
        return '\\mathcal{{P}}\\left({}\\right)'.format(arg_print)

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return '\\left\\{%s\\; \\middle|\\; %s \\in %s \\right\\}' % (self._print(s.expr), vars_print, self._print(s.sets))

    def _print_Contains(self, e):
        return '%s \\in %s' % tuple((self._print(a) for a in e.args))

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        return self._print_Add(s.truncate()) + ' + \\ldots'

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return '\\mathbb{F}_{%s}' % expr.mod

    def _print_IntegerRing(self, expr):
        return '\\mathbb{Z}'

    def _print_RationalField(self, expr):
        return '\\mathbb{Q}'

    def _print_RealField(self, expr):
        return '\\mathbb{R}'

    def _print_ComplexField(self, expr):
        return '\\mathbb{C}'

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ', '.join(map(self._print, expr.symbols))
        return '%s\\left[%s\\right]' % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ', '.join(map(self._print, expr.symbols))
        return '%s\\left(%s\\right)' % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ', '.join(map(self._print, expr.symbols))
        inv = ''
        if not expr.is_Poly:
            inv = 'S_<^{-1}'
        return '%s%s\\left[%s\\right]' % (inv, domain, symbols)

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))
            if coeff.is_Add:
                if s_monom:
                    s_coeff = '\\left(%s\\right)' % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue
                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue
                s_coeff = self._print(coeff)
            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + ' ' + s_monom
            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])
        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)
            if modifier == '-':
                terms[0] = '-' + terms[0]
        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = 'domain=%s' % self._print(poly.get_domain())
        args = ', '.join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = '\\%s {\\left(%s \\right)}' % (cls, args)
        else:
            tex = '\\operatorname{%s}{\\left( %s \\right)}' % (cls, args)
        return tex

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == 'ComplexRootOf':
            cls = 'CRootOf'
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return '\\%s {\\left(%s, %d\\right)}' % (cls, expr, index)
        else:
            return '\\operatorname{%s} {\\left(%s, %d\\right)}' % (cls, expr, index)

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]
        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))
        if cls in accepted_latex_functions:
            return '\\%s {\\left(%s\\right)}' % (cls, ', '.join(args))
        else:
            return '\\operatorname{%s} {\\left(%s\\right)}' % (cls, ', '.join(args))

    def _print_OrdinalOmega(self, expr):
        return '\\omega'

    def _print_OmegaPower(self, expr):
        exp, mul = expr.args
        if mul != 1:
            if exp != 1:
                return '{} \\omega^{{{}}}'.format(mul, exp)
            else:
                return '{} \\omega'.format(mul)
        elif exp != 1:
            return '\\omega^{{{}}}'.format(exp)
        else:
            return '\\omega'

    def _print_Ordinal(self, expr):
        return ' + '.join([self._print(arg) for arg in expr.args])

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, '{%s}^{%d}', mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return '\\frac{%s}{%s}' % (numer, denom)

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = 'E_{%s}' % self._print(m)
        if exp is not None:
            tex = '%s^{%s}' % (tex, exp)
        if x is not None:
            tex = '%s\\left(%s\\right)' % (tex, self._print(x))
        return tex

    def _print_catalan(self, expr, exp=None):
        tex = 'C_{%s}' % self._print(expr.args[0])
        if exp is not None:
            tex = '%s^{%s}' % (tex, exp)
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return '\\mathcal{{{}}}{}_{{{}}}\\left[{}\\right]\\left({}\\right)'.format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr, exp=None):
        arg0 = self._print(expr.args[0])
        exp = '^{%s}' % (exp,) if exp is not None else ''
        if len(expr.args) == 1:
            result = 'W%s\\left(%s\\right)' % (exp, arg0)
        else:
            arg1 = self._print(expr.args[1])
            result = 'W{0}_{{{1}}}\\left({2}\\right)'.format(exp, arg1, arg0)
        return result

    def _print_Expectation(self, expr):
        return '\\operatorname{{E}}\\left[{}\\right]'.format(self._print(expr.args[0]))

    def _print_Variance(self, expr):
        return '\\operatorname{{Var}}\\left({}\\right)'.format(self._print(expr.args[0]))

    def _print_Covariance(self, expr):
        return '\\operatorname{{Cov}}\\left({}\\right)'.format(', '.join((self._print(arg) for arg in expr.args)))

    def _print_Probability(self, expr):
        return '\\operatorname{{P}}\\left({}\\right)'.format(self._print(expr.args[0]))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return '%s\\rightarrow %s' % (domain, codomain)

    def _print_TransferFunction(self, expr):
        num, den = (self._print(expr.num), self._print(expr.den))
        return '\\frac{%s}{%s}' % (num, den)

    def _print_Series(self, expr):
        args = list(expr.args)
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr), False)
        return ' '.join(map(parens, args))

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)[::-1]
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr), False) if isinstance(x, MIMOParallel) else self._print(x)
        return '\\cdot'.join(map(parens, args))

    def _print_Parallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_MIMOParallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series
        num, tf = (expr.sys1, TransferFunction(1, 1, expr.var))
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if isinstance(expr.sys2, Series) else [expr.sys2]
        den_term_1 = tf
        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = (tf, Series(*num_arg_list, expr.sys2))
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            else:
                den_term_2 = Series(num, *den_arg_list)
        elif num == tf:
            den_term_2 = Series(*den_arg_list)
        elif expr.sys2 == tf:
            den_term_2 = Series(*num_arg_list)
        else:
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        numer = self._print(num)
        denom_1 = self._print(den_term_1)
        denom_2 = self._print(den_term_2)
        _sign = '+' if expr.sign == -1 else '-'
        return '\\frac{%s}{%s %s %s}' % (numer, denom_1, _sign, denom_2)

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        sys1 = self._print(expr.sys1)
        _sign = '+' if expr.sign == -1 else '-'
        return '\\left(I_{\\tau} %s %s\\right)^{-1} \\cdot %s' % (_sign, inv_mat, sys1)

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        return '%s_\\tau' % mat

    def _print_DFT(self, expr):
        return '\\text{{{}}}_{{{}}}'.format(expr.__class__.__name__, expr.n)
    _print_IDFT = _print_DFT

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return '%s:%s' % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(morphism.domain, morphism.codomain, 'id'))

    def _print_CompositeMorphism(self, morphism):
        component_names_list = [self._print(Symbol(component.name)) for component in morphism.components]
        component_names_list.reverse()
        component_names = '\\circ '.join(component_names_list) + ':'
        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return '\\mathbf{{{}}}'.format(self._print(Symbol(morphism.name)))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            return self._print(S.EmptySet)
        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += '\\Longrightarrow %s' % self._print(diagram.conclusions)
        return latex_result

    def _print_DiagramGrid(self, grid):
        latex_result = '\\begin{array}{%s}\n' % ('c' * grid.width)
        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += ' '
                if j != grid.width - 1:
                    latex_result += '& '
            if i != grid.height - 1:
                latex_result += '\\\\'
            latex_result += '\n'
        latex_result += '\\end{array}\n'
        return latex_result

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        return '\\left[ {} \\right]'.format(','.join(('{' + self._print(x) + '}' for x in m)))

    def _print_SubModule(self, m):
        return '\\left\\langle {} \\right\\rangle'.format(','.join(('{' + self._print(x) + '}' for x in m.gens)))

    def _print_ModuleImplementedIdeal(self, m):
        return '\\left\\langle {} \\right\\rangle'.format(','.join(('{' + self._print(x) + '}' for [x] in m._module.gens)))

    def _print_Quaternion(self, expr):
        s = [self.parenthesize(i, PRECEDENCE['Mul'], strict=True) for i in expr.args]
        a = [s[0]] + [i + ' ' + j for i, j in zip(s[1:], 'ijk')]
        return ' + '.join(a)

    def _print_QuotientRing(self, R):
        return '\\frac{{{}}}{{{}}}'.format(self._print(R.ring), self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        return '{{{}}} + {{{}}}'.format(self._print(x.data), self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        return '{{{}}} + {{{}}}'.format(self._print(m.data), self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        return '\\frac{{{}}}{{{}}}'.format(self._print(M.base), self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return '{{{}}} : {{{}}} \\to {{{}}}'.format(self._print(h._sympy_matrix()), self._print(h.domain), self._print(h.codomain))

    def _print_Manifold(self, manifold):
        string = manifold.name.name
        if '{' in string:
            name, supers, subs = (string, [], [])
        else:
            name, supers, subs = split_super_sub(string)
            name = translate(name)
            supers = [translate(sup) for sup in supers]
            subs = [translate(sub) for sub in subs]
        name = '\\text{%s}' % name
        if supers:
            name += '^{%s}' % ' '.join(supers)
        if subs:
            name += '_{%s}' % ' '.join(subs)
        return name

    def _print_Patch(self, patch):
        return '\\text{%s}_{%s}' % (self._print(patch.name), self._print(patch.manifold))

    def _print_CoordSystem(self, coordsys):
        return '\\text{%s}^{\\text{%s}}_{%s}' % (self._print(coordsys.name), self._print(coordsys.patch.name), self._print(coordsys.manifold))

    def _print_CovarDerivativeOp(self, cvd):
        return '\\mathbb{\\nabla}_{%s}' % self._print(cvd._wrt)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return '\\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return '\\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return '\\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return '\\operatorname{{d}}\\left({}\\right)'.format(string)

    def _print_Tr(self, p):
        contents = self._print(p.args[0])
        return '\\operatorname{{tr}}\\left({}\\right)'.format(contents)

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return '\\left(\\phi\\left(%s\\right)\\right)^{%s}' % (self._print(expr.args[0]), exp)
        return '\\phi\\left(%s\\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return '\\left(\\lambda\\left(%s\\right)\\right)^{%s}' % (self._print(expr.args[0]), exp)
        return '\\lambda\\left(%s\\right)' % self._print(expr.args[0])

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = '_%s\\left(%s\\right)' % tuple(map(self._print, (expr.args[1], expr.args[0])))
        else:
            tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '\\sigma^{%s}%s' % (exp, tex)
        return '\\sigma%s' % tex

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = '_%s\\left(%s\\right)' % tuple(map(self._print, (expr.args[1], expr.args[0])))
        else:
            tex = '\\left(%s\\right)' % self._print(expr.args[0])
        if exp is not None:
            return '\\sigma^*^{%s}%s' % (exp, tex)
        return '\\sigma^*%s' % tex

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return '\\left(\\nu\\left(%s\\right)\\right)^{%s}' % (self._print(expr.args[0]), exp)
        return '\\nu\\left(%s\\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return '\\left(\\Omega\\left(%s\\right)\\right)^{%s}' % (self._print(expr.args[0]), exp)
        return '\\Omega\\left(%s\\right)' % self._print(expr.args[0])

    def _print_Str(self, s):
        return str(s.name)

    def _print_float(self, expr):
        return self._print(Float(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_mpq(self, expr):
        return str(expr)

    def _print_Predicate(self, expr):
        return '\\operatorname{{Q}}_{{\\text{{{}}}}}'.format(latex_escape(str(expr.name)))

    def _print_AppliedPredicate(self, expr):
        pred = expr.function
        args = expr.arguments
        pred_latex = self._print(pred)
        args_latex = ', '.join([self._print(a) for a in args])
        return '%s(%s)' % (pred_latex, args_latex)

    def emptyPrinter(self, expr):
        s = super().emptyPrinter(expr)
        return '\\mathtt{\\text{%s}}' % latex_escape(s)