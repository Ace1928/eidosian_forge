from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
class VectorStrPrinter(StrPrinter):
    """String Printer for vector expressions. """

    def _print_Derivative(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if bool(sum([i == t for i in e.variables])) & isinstance(type(e.args[0]), UndefinedFunction):
            ol = str(e.args[0].func)
            for i, v in enumerate(e.variables):
                ol += dynamicsymbols._str
            return ol
        else:
            return StrPrinter().doprint(e)

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if isinstance(type(e), UndefinedFunction):
            return StrPrinter().doprint(e).replace('(%s)' % t, '')
        return e.func.__name__ + '(%s)' % self.stringify(e.args, ', ')