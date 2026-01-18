from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
class CodePrinter(StrPrinter):
    """
    The base class for code-printing subclasses.
    """
    _operators = {'and': '&&', 'or': '||', 'not': '!'}
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'error_on_reserved': False, 'reserved_word_suffix': '_', 'human': True, 'inline': False, 'allow_unknown_functions': False}
    _rewriteable_functions = {'cot': ('tan', []), 'csc': ('sin', []), 'sec': ('cos', []), 'acot': ('atan', []), 'acsc': ('asin', []), 'asec': ('acos', []), 'coth': ('exp', []), 'csch': ('exp', []), 'sech': ('exp', []), 'acoth': ('log', []), 'acsch': ('log', []), 'asech': ('log', []), 'catalan': ('gamma', []), 'fibonacci': ('sqrt', []), 'lucas': ('sqrt', []), 'beta': ('gamma', []), 'sinc': ('sin', ['Piecewise']), 'Mod': ('floor', []), 'factorial': ('gamma', []), 'factorial2': ('gamma', ['Piecewise']), 'subfactorial': ('uppergamma', []), 'RisingFactorial': ('gamma', ['Piecewise']), 'FallingFactorial': ('gamma', ['Piecewise']), 'binomial': ('gamma', []), 'frac': ('floor', []), 'Max': ('Piecewise', []), 'Min': ('Piecewise', []), 'Heaviside': ('Piecewise', []), 'erf2': ('erf', []), 'erfc': ('erf', []), 'Li': ('li', []), 'Ei': ('li', []), 'dirichlet_eta': ('zeta', []), 'riemann_xi': ('zeta', ['gamma'])}

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        if not hasattr(self, 'reserved_words'):
            self.reserved_words = set()

    def _handle_UnevaluatedExpr(self, expr):
        return expr.replace(re, lambda arg: arg if isinstance(arg, UnevaluatedExpr) and arg.args[0].is_real else re(arg))

    def doprint(self, expr, assign_to=None):
        """
        Print the expression as code.

        Parameters
        ----------
        expr : Expression
            The expression to be printed.

        assign_to : Symbol, string, MatrixSymbol, list of strings or Symbols (optional)
            If provided, the printed code will set the expression to a variable or multiple variables
            with the name or names given in ``assign_to``.
        """
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.codegen.ast import CodeBlock, Assignment

        def _handle_assign_to(expr, assign_to):
            if assign_to is None:
                return sympify(expr)
            if isinstance(assign_to, (list, tuple)):
                if len(expr) != len(assign_to):
                    raise ValueError('Failed to assign an expression of length {} to {} variables'.format(len(expr), len(assign_to)))
                return CodeBlock(*[_handle_assign_to(lhs, rhs) for lhs, rhs in zip(expr, assign_to)])
            if isinstance(assign_to, str):
                if expr.is_Matrix:
                    assign_to = MatrixSymbol(assign_to, *expr.shape)
                else:
                    assign_to = Symbol(assign_to)
            elif not isinstance(assign_to, Basic):
                raise TypeError('{} cannot assign to object of type {}'.format(type(self).__name__, type(assign_to)))
            return Assignment(assign_to, expr)
        expr = _convert_python_lists(expr)
        expr = _handle_assign_to(expr, assign_to)
        expr = self._handle_UnevaluatedExpr(expr)
        self._not_supported = set()
        self._number_symbols = set()
        lines = self._print(expr).splitlines()
        if self._settings['human']:
            frontlines = []
            if self._not_supported:
                frontlines.append(self._get_comment('Not supported in {}:'.format(self.language)))
                for expr in sorted(self._not_supported, key=str):
                    frontlines.append(self._get_comment(type(expr).__name__))
            for name, value in sorted(self._number_symbols, key=str):
                frontlines.append(self._declare_number_const(name, value))
            lines = frontlines + lines
            lines = self._format_code(lines)
            result = '\n'.join(lines)
        else:
            lines = self._format_code(lines)
            num_syms = {(k, self._print(v)) for k, v in self._number_symbols}
            result = (num_syms, self._not_supported, '\n'.join(lines))
        self._not_supported = set()
        self._number_symbols = set()
        return result

    def _doprint_loops(self, expr, assign_to=None):
        if self._settings.get('contract', True):
            from sympy.tensor import get_contraction_structure
            indices = self._get_expression_indices(expr, assign_to)
            dummies = get_contraction_structure(expr)
        else:
            indices = []
            dummies = {None: (expr,)}
        openloop, closeloop = self._get_loop_opening_ending(indices)
        if None in dummies:
            text = StrPrinter.doprint(self, Add(*dummies[None]))
        else:
            text = StrPrinter.doprint(self, 0)
        lhs_printed = self._print(assign_to)
        lines = []
        if text != lhs_printed:
            lines.extend(openloop)
            if assign_to is not None:
                text = self._get_statement('%s = %s' % (lhs_printed, text))
            lines.append(text)
            lines.extend(closeloop)
        for d in dummies:
            if isinstance(d, tuple):
                indices = self._sort_optimized(d, expr)
                openloop_d, closeloop_d = self._get_loop_opening_ending(indices)
                for term in dummies[d]:
                    if term in dummies and (not [list(f.keys()) for f in dummies[term]] == [[None] for f in dummies[term]]):
                        raise NotImplementedError('FIXME: no support for contractions in factor yet')
                    else:
                        if assign_to is None:
                            raise AssignmentError('need assignment variable for loops')
                        if term.has(assign_to):
                            raise ValueError('FIXME: lhs present in rhs,                                this is undefined in CodePrinter')
                        lines.extend(openloop)
                        lines.extend(openloop_d)
                        text = '%s = %s' % (lhs_printed, StrPrinter.doprint(self, assign_to + term))
                        lines.append(self._get_statement(text))
                        lines.extend(closeloop_d)
                        lines.extend(closeloop)
        return '\n'.join(lines)

    def _get_expression_indices(self, expr, assign_to):
        from sympy.tensor import get_indices
        rinds, junk = get_indices(expr)
        linds, junk = get_indices(assign_to)
        if linds and (not rinds):
            rinds = linds
        if rinds != linds:
            raise ValueError('lhs indices must match non-dummy rhs indices in %s' % expr)
        return self._sort_optimized(rinds, assign_to)

    def _sort_optimized(self, indices, expr):
        from sympy.tensor.indexed import Indexed
        if not indices:
            return []
        score_table = {}
        for i in indices:
            score_table[i] = 0
        arrays = expr.atoms(Indexed)
        for arr in arrays:
            for p, ind in enumerate(arr.indices):
                try:
                    score_table[ind] += self._rate_index_position(p)
                except KeyError:
                    pass
        return sorted(indices, key=lambda x: score_table[x])

    def _rate_index_position(self, p):
        """function to calculate score based on position among indices

        This method is used to sort loops in an optimized order, see
        CodePrinter._sort_optimized()
        """
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_statement(self, codestring):
        """Formats a codestring with the proper line ending."""
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_comment(self, text):
        """Formats a text string as a comment."""
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _declare_number_const(self, name, value):
        """Declare a numeric constant at the top of a function"""
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _format_code(self, lines):
        """Take in a list of lines of code, and format them accordingly.

        This may include indenting, wrapping long lines, etc..."""
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists
        of codelines"""
        raise NotImplementedError('This function must be implemented by subclass of CodePrinter.')

    def _print_Dummy(self, expr):
        if expr.name.startswith('Dummy_'):
            return '_' + expr.name
        else:
            return '%s_%d' % (expr.name, expr.dummy_index)

    def _print_CodeBlock(self, expr):
        return '\n'.join([self._print(i) for i in expr.args])

    def _print_String(self, string):
        return str(string)

    def _print_QuotedString(self, arg):
        return '"%s"' % arg.text

    def _print_Comment(self, string):
        return self._get_comment(str(string))

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(expr.rhs, Piecewise):
            expressions = []
            conditions = []
            for e, c in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        elif isinstance(lhs, MatrixSymbol):
            lines = []
            for i, j in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return '\n'.join(lines)
        elif self._settings.get('contract', False) and (lhs.has(IndexedBase) or rhs.has(IndexedBase)):
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement('%s = %s' % (lhs_code, rhs_code))

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        return self._get_statement('{} {} {}'.format(*(self._print(arg) for arg in [lhs_code, expr.op, rhs_code])))

    def _print_FunctionCall(self, expr):
        return '%s(%s)' % (expr.name, ', '.join((self._print(arg) for arg in expr.function_args)))

    def _print_Variable(self, expr):
        return self._print(expr.symbol)

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = 'This expression includes the symbol "{}" which is a reserved keyword in this language.'
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        else:
            return name

    def _can_print(self, name):
        """ Check if function ``name`` is either a known function or has its own
            printing method. Used to check if rewriting is possible."""
        return name in self.known_functions or getattr(self, '_print_{}'.format(name), False)

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_func = self.known_functions[expr.func.__name__]
            if isinstance(cond_func, str):
                return '%s(%s)' % (cond_func, self.stringify(expr.args, ', '))
            else:
                for cond, func in cond_func:
                    if cond(*expr.args):
                        break
                if func is not None:
                    try:
                        return func(*[self.parenthesize(item, 0) for item in expr.args])
                    except TypeError:
                        return '%s(%s)' % (func, self.stringify(expr.args, ', '))
        elif hasattr(expr, '_imp_') and isinstance(expr._imp_, Lambda):
            return self._print(expr._imp_(*expr.args))
        elif expr.func.__name__ in self._rewriteable_functions:
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all((self._can_print(f) for f in required_fs)):
                return self._print(expr.rewrite(target_f))
        if expr.is_Function and self._settings.get('allow_unknown_functions', False):
            return '%s(%s)' % (self._print(expr.func), ', '.join(map(self._print, expr.args)))
        else:
            return self._print_not_supported(expr)
    _print_Expr = _print_Function
    _print_Heaviside = None

    def _print_NumberSymbol(self, expr):
        if self._settings.get('inline', False):
            return self._print(Float(expr.evalf(self._settings['precision'])))
        else:
            self._number_symbols.add((expr, Float(expr.evalf(self._settings['precision']))))
            return str(expr)

    def _print_Catalan(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_EulerGamma(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_GoldenRatio(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_TribonacciConstant(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_Exp1(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_Pi(self, expr):
        return self._print_NumberSymbol(expr)

    def _print_And(self, expr):
        PREC = precedence(expr)
        return (' %s ' % self._operators['and']).join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return (' %s ' % self._operators['or']).join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

    def _print_Xor(self, expr):
        if self._operators.get('xor') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (' %s ' % self._operators['xor']).join((self.parenthesize(a, PREC) for a in expr.args))

    def _print_Equivalent(self, expr):
        if self._operators.get('equivalent') is None:
            return self._print(expr.to_nnf())
        PREC = precedence(expr)
        return (' %s ' % self._operators['equivalent']).join((self.parenthesize(a, PREC) for a in expr.args))

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + self.parenthesize(expr.args[0], PREC)

    def _print_BooleanFunction(self, expr):
        return self._print(expr.to_nnf())

    def _print_Mul(self, expr):
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
            else:
                a.append(item)
        a = a or [S.One]
        if len(a) == 1 and sign == '-':
            a_str = [self.parenthesize(a[0], 0.5 * (PRECEDENCE['Pow'] + PRECEDENCE['Mul']))]
        else:
            a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]
        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + '/' + b_str[0]
        else:
            return sign + '*'.join(a_str) + '/(%s)' % '*'.join(b_str)

    def _print_not_supported(self, expr):
        try:
            self._not_supported.add(expr)
        except TypeError:
            pass
        return self.emptyPrinter(expr)
    _print_Basic = _print_not_supported
    _print_ComplexInfinity = _print_not_supported
    _print_Derivative = _print_not_supported
    _print_ExprCondPair = _print_not_supported
    _print_GeometryEntity = _print_not_supported
    _print_Infinity = _print_not_supported
    _print_Integral = _print_not_supported
    _print_Interval = _print_not_supported
    _print_AccumulationBounds = _print_not_supported
    _print_Limit = _print_not_supported
    _print_MatrixBase = _print_not_supported
    _print_DeferredVector = _print_not_supported
    _print_NaN = _print_not_supported
    _print_NegativeInfinity = _print_not_supported
    _print_Order = _print_not_supported
    _print_RootOf = _print_not_supported
    _print_RootsOf = _print_not_supported
    _print_RootSum = _print_not_supported
    _print_Uniform = _print_not_supported
    _print_Unit = _print_not_supported
    _print_Wild = _print_not_supported
    _print_WildFunction = _print_not_supported
    _print_Relational = _print_not_supported