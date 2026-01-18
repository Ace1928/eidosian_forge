from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
class OuterProduct(Operator):
    """An unevaluated outer product between a ket and bra.

    This constructs an outer product between any subclass of ``KetBase`` and
    ``BraBase`` as ``|a><b|``. An ``OuterProduct`` inherits from Operator as they act as
    operators in quantum expressions.  For reference see [1]_.

    Parameters
    ==========

    ket : KetBase
        The ket on the left side of the outer product.
    bar : BraBase
        The bra on the right side of the outer product.

    Examples
    ========

    Create a simple outer product by hand and take its dagger::

        >>> from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger
        >>> from sympy.physics.quantum import Operator

        >>> k = Ket('k')
        >>> b = Bra('b')
        >>> op = OuterProduct(k, b)
        >>> op
        |k><b|
        >>> op.hilbert_space
        H
        >>> op.ket
        |k>
        >>> op.bra
        <b|
        >>> Dagger(op)
        |b><k|

    In simple products of kets and bras outer products will be automatically
    identified and created::

        >>> k*b
        |k><b|

    But in more complex expressions, outer products are not automatically
    created::

        >>> A = Operator('A')
        >>> A*k*b
        A*|k>*<b|

    A user can force the creation of an outer product in a complex expression
    by using parentheses to group the ket and bra::

        >>> A*(k*b)
        A*|k><b|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Outer_product
    """
    is_commutative = False

    def __new__(cls, *args, **old_assumptions):
        from sympy.physics.quantum.state import KetBase, BraBase
        if len(args) != 2:
            raise ValueError('2 parameters expected, got %d' % len(args))
        ket_expr = expand(args[0])
        bra_expr = expand(args[1])
        if isinstance(ket_expr, (KetBase, Mul)) and isinstance(bra_expr, (BraBase, Mul)):
            ket_c, kets = ket_expr.args_cnc()
            bra_c, bras = bra_expr.args_cnc()
            if len(kets) != 1 or not isinstance(kets[0], KetBase):
                raise TypeError('KetBase subclass expected, got: %r' % Mul(*kets))
            if len(bras) != 1 or not isinstance(bras[0], BraBase):
                raise TypeError('BraBase subclass expected, got: %r' % Mul(*bras))
            if not kets[0].dual_class() == bras[0].__class__:
                raise TypeError('ket and bra are not dual classes: %r, %r' % (kets[0].__class__, bras[0].__class__))
            obj = Expr.__new__(cls, *(kets[0], bras[0]), **old_assumptions)
            obj.hilbert_space = kets[0].hilbert_space
            return Mul(*ket_c + bra_c) * obj
        op_terms = []
        if isinstance(ket_expr, Add) and isinstance(bra_expr, Add):
            for ket_term in ket_expr.args:
                for bra_term in bra_expr.args:
                    op_terms.append(OuterProduct(ket_term, bra_term, **old_assumptions))
        elif isinstance(ket_expr, Add):
            for ket_term in ket_expr.args:
                op_terms.append(OuterProduct(ket_term, bra_expr, **old_assumptions))
        elif isinstance(bra_expr, Add):
            for bra_term in bra_expr.args:
                op_terms.append(OuterProduct(ket_expr, bra_term, **old_assumptions))
        else:
            raise TypeError('Expected ket and bra expression, got: %r, %r' % (ket_expr, bra_expr))
        return Add(*op_terms)

    @property
    def ket(self):
        """Return the ket on the left side of the outer product."""
        return self.args[0]

    @property
    def bra(self):
        """Return the bra on the right side of the outer product."""
        return self.args[1]

    def _eval_adjoint(self):
        return OuterProduct(Dagger(self.bra), Dagger(self.ket))

    def _sympystr(self, printer, *args):
        return printer._print(self.ket) + printer._print(self.bra)

    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__, printer._print(self.ket, *args), printer._print(self.bra, *args))

    def _pretty(self, printer, *args):
        pform = self.ket._pretty(printer, *args)
        return prettyForm(*pform.right(self.bra._pretty(printer, *args)))

    def _latex(self, printer, *args):
        k = printer._print(self.ket, *args)
        b = printer._print(self.bra, *args)
        return k + b

    def _represent(self, **options):
        k = self.ket._represent(**options)
        b = self.bra._represent(**options)
        return k * b

    def _eval_trace(self, **kwargs):
        return self.ket._eval_trace(self.bra, **kwargs)