import typing
import sympy
from sympy.core import Add, Mul
from sympy.core import Symbol, Expr, Float, Rational, Integer, Basic
from sympy.core.function import UndefinedFunction, Function
from sympy.core.relational import Relational, Unequality, Equality, LessThan, GreaterThan, StrictLessThan, StrictGreaterThan
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log, Pow
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, tan, asin, acos, atan, atan2
from sympy.logic.boolalg import And, Or, Xor, Implies, Boolean
from sympy.logic.boolalg import BooleanTrue, BooleanFalse, BooleanFunction, Not, ITE
from sympy.printing.printer import Printer
from sympy.sets import Interval
class SMTLibPrinter(Printer):
    printmethod = '_smtlib'
    _default_settings: dict = {'precision': None, 'known_types': {bool: 'Bool', int: 'Int', float: 'Real'}, 'known_constants': {}, 'known_functions': {Add: '+', Mul: '*', Equality: '=', LessThan: '<=', GreaterThan: '>=', StrictLessThan: '<', StrictGreaterThan: '>', exp: 'exp', log: 'log', Abs: 'abs', sin: 'sin', cos: 'cos', tan: 'tan', asin: 'arcsin', acos: 'arccos', atan: 'arctan', atan2: 'arctan2', sinh: 'sinh', cosh: 'cosh', tanh: 'tanh', Min: 'min', Max: 'max', Pow: 'pow', And: 'and', Or: 'or', Xor: 'xor', Not: 'not', ITE: 'ite', Implies: '=>'}}
    symbol_table: dict

    def __init__(self, settings: typing.Optional[dict]=None, symbol_table=None):
        settings = settings or {}
        self.symbol_table = symbol_table or {}
        Printer.__init__(self, settings)
        self._precision = self._settings['precision']
        self._known_types = dict(self._settings['known_types'])
        self._known_constants = dict(self._settings['known_constants'])
        self._known_functions = dict(self._settings['known_functions'])
        for _ in self._known_types.values():
            assert self._is_legal_name(_)
        for _ in self._known_constants.values():
            assert self._is_legal_name(_)

    def _is_legal_name(self, s: str):
        if not s:
            return False
        if s[0].isnumeric():
            return False
        return all((_.isalnum() or _ == '_' for _ in s))

    def _s_expr(self, op: str, args: typing.Union[list, tuple]) -> str:
        args_str = ' '.join((a if isinstance(a, str) else self._print(a) for a in args))
        return f'({op} {args_str})'

    def _print_Function(self, e):
        if e in self._known_functions:
            op = self._known_functions[e]
        elif type(e) in self._known_functions:
            op = self._known_functions[type(e)]
        elif type(type(e)) == UndefinedFunction:
            op = e.name
        else:
            op = self._known_functions[e]
        return self._s_expr(op, e.args)

    def _print_Relational(self, e: Relational):
        return self._print_Function(e)

    def _print_BooleanFunction(self, e: BooleanFunction):
        return self._print_Function(e)

    def _print_Expr(self, e: Expr):
        return self._print_Function(e)

    def _print_Unequality(self, e: Unequality):
        if type(e) in self._known_functions:
            return self._print_Relational(e)
        else:
            eq_op = self._known_functions[Equality]
            not_op = self._known_functions[Not]
            return self._s_expr(not_op, [self._s_expr(eq_op, e.args)])

    def _print_Piecewise(self, e: Piecewise):

        def _print_Piecewise_recursive(args: typing.Union[list, tuple]):
            e, c = args[0]
            if len(args) == 1:
                assert c is True or isinstance(c, BooleanTrue)
                return self._print(e)
            else:
                ite = self._known_functions[ITE]
                return self._s_expr(ite, [c, e, _print_Piecewise_recursive(args[1:])])
        return _print_Piecewise_recursive(e.args)

    def _print_Interval(self, e: Interval):
        if e.start.is_infinite and e.end.is_infinite:
            return ''
        elif e.start.is_infinite != e.end.is_infinite:
            raise ValueError(f'One-sided intervals (`{e}`) are not supported in SMT.')
        else:
            return f'[{e.start}, {e.end}]'

    def _print_BooleanTrue(self, x: BooleanTrue):
        return 'true'

    def _print_BooleanFalse(self, x: BooleanFalse):
        return 'false'

    def _print_Float(self, x: Float):
        f = x.evalf(self._precision) if self._precision else x.evalf()
        return str(f).rstrip('0')

    def _print_float(self, x: float):
        return str(x)

    def _print_Rational(self, x: Rational):
        return self._s_expr('/', [x.p, x.q])

    def _print_Integer(self, x: Integer):
        assert x.q == 1
        return str(x.p)

    def _print_int(self, x: int):
        return str(x)

    def _print_Symbol(self, x: Symbol):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_NumberSymbol(self, x):
        name = self._known_constants.get(x)
        return name if name else self._print_Float(x)

    def _print_UndefinedFunction(self, x):
        assert self._is_legal_name(x.name)
        return x.name

    def _print_Exp1(self, x):
        return self._print_Function(exp(1, evaluate=False)) if exp in self._known_functions else self._print_NumberSymbol(x)

    def emptyPrinter(self, expr):
        raise NotImplementedError(f'Cannot convert `{repr(expr)}` of type `{type(expr)}` to SMT.')