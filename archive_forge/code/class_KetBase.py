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
class KetBase(StateBase):
    """Base class for Kets.

    This class defines the dual property and the brackets for printing. This is
    an abstract base class and you should not instantiate it directly, instead
    use Ket.
    """
    lbracket = _straight_bracket
    rbracket = _rbracket
    lbracket_ucode = _straight_bracket_ucode
    rbracket_ucode = _rbracket_ucode
    lbracket_latex = '\\left|'
    rbracket_latex = '\\right\\rangle '

    @classmethod
    def default_args(self):
        return ('psi',)

    @classmethod
    def dual_class(self):
        return BraBase

    def __mul__(self, other):
        """KetBase*other"""
        from sympy.physics.quantum.operator import OuterProduct
        if isinstance(other, BraBase):
            return OuterProduct(self, other)
        else:
            return Expr.__mul__(self, other)

    def __rmul__(self, other):
        """other*KetBase"""
        from sympy.physics.quantum.innerproduct import InnerProduct
        if isinstance(other, BraBase):
            return InnerProduct(other, self)
        else:
            return Expr.__rmul__(self, other)

    def _eval_innerproduct(self, bra, **hints):
        """Evaluate the inner product between this ket and a bra.

        This is called to compute <bra|ket>, where the ket is ``self``.

        This method will dispatch to sub-methods having the format::

            ``def _eval_innerproduct_BraClass(self, **hints):``

        Subclasses should define these methods (one for each BraClass) to
        teach the ket how to take inner products with bras.
        """
        return dispatch_method(self, '_eval_innerproduct', bra, **hints)

    def _apply_from_right_to(self, op, **options):
        """Apply an Operator to this Ket as Operator*Ket

        This method will dispatch to methods having the format::

            ``def _apply_from_right_to_OperatorName(op, **options):``

        Subclasses should define these methods (one for each OperatorName) to
        teach the Ket how to implement OperatorName*Ket

        Parameters
        ==========

        op : Operator
            The Operator that is acting on the Ket as op*Ket
        options : dict
            A dict of key/value pairs that control how the operator is applied
            to the Ket.
        """
        return dispatch_method(self, '_apply_from_right_to', op, **options)