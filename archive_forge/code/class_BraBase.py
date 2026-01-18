from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
class BraBase(StateBase):
    """Base class for Bras.

    This class defines the dual property and the brackets for printing. This
    is an abstract base class and you should not instantiate it directly,
    instead use Bra.
    """
    lbracket = _lbracket
    rbracket = _straight_bracket
    lbracket_ucode = _lbracket_ucode
    rbracket_ucode = _straight_bracket_ucode
    lbracket_latex = '\\left\\langle '
    rbracket_latex = '\\right|'

    @classmethod
    def _operators_to_state(self, ops, **options):
        state = self.dual_class()._operators_to_state(ops, **options)
        return state.dual

    def _state_to_operators(self, op_classes, **options):
        return self.dual._state_to_operators(op_classes, **options)

    def _enumerate_state(self, num_states, **options):
        dual_states = self.dual._enumerate_state(num_states, **options)
        return [x.dual for x in dual_states]

    @classmethod
    def default_args(self):
        return self.dual_class().default_args()

    @classmethod
    def dual_class(self):
        return KetBase

    def __mul__(self, other):
        """BraBase*other"""
        from sympy.physics.quantum.innerproduct import InnerProduct
        if isinstance(other, KetBase):
            return InnerProduct(self, other)
        else:
            return Expr.__mul__(self, other)

    def __rmul__(self, other):
        """other*BraBase"""
        from sympy.physics.quantum.operator import OuterProduct
        if isinstance(other, KetBase):
            return OuterProduct(other, self)
        else:
            return Expr.__rmul__(self, other)

    def _represent(self, **options):
        """A default represent that uses the Ket's version."""
        from sympy.physics.quantum.dagger import Dagger
        return Dagger(self.dual._represent(**options))