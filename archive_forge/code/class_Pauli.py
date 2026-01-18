from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.symbol import Symbol
from sympy.physics.quantum import TensorProduct
class Pauli(Symbol):
    """
    The class representing algebraic properties of Pauli matrices.

    Explanation
    ===========

    The symbol used to display the Pauli matrices can be changed with an
    optional parameter ``label="sigma"``. Pauli matrices with different
    ``label`` attributes cannot multiply together.

    If the left multiplication of symbol or number with Pauli matrix is needed,
    please use parentheses  to separate Pauli and symbolic multiplication
    (for example: 2*I*(Pauli(3)*Pauli(2))).

    Another variant is to use evaluate_pauli_product function to evaluate
    the product of Pauli matrices and other symbols (with commutative
    multiply rules).

    See Also
    ========

    evaluate_pauli_product

    Examples
    ========

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1)
    sigma1
    >>> Pauli(1)*Pauli(2)
    I*sigma3
    >>> Pauli(1)*Pauli(1)
    1
    >>> Pauli(3)**4
    1
    >>> Pauli(1)*Pauli(2)*Pauli(3)
    I

    >>> from sympy.physics.paulialgebra import Pauli
    >>> Pauli(1, label="tau")
    tau1
    >>> Pauli(1)*Pauli(2, label="tau")
    sigma1*tau2
    >>> Pauli(1, label="tau")*Pauli(2, label="tau")
    I*tau3

    >>> from sympy import I
    >>> I*(Pauli(2)*Pauli(3))
    -sigma1

    >>> from sympy.physics.paulialgebra import evaluate_pauli_product
    >>> f = I*Pauli(2)*Pauli(3)
    >>> f
    I*sigma2*sigma3
    >>> evaluate_pauli_product(f)
    -sigma1
    """
    __slots__ = ('i', 'label')

    def __new__(cls, i, label='sigma'):
        if i not in [1, 2, 3]:
            raise IndexError('Invalid Pauli index')
        obj = Symbol.__new__(cls, '%s%d' % (label, i), commutative=False, hermitian=True)
        obj.i = i
        obj.label = label
        return obj

    def __getnewargs_ex__(self):
        return ((self.i, self.label), {})

    def _hashable_content(self):
        return (self.i, self.label)

    def __mul__(self, other):
        if isinstance(other, Pauli):
            j = self.i
            k = other.i
            jlab = self.label
            klab = other.label
            if jlab == klab:
                return delta(j, k) + I * epsilon(j, k, 1) * Pauli(1, jlab) + I * epsilon(j, k, 2) * Pauli(2, jlab) + I * epsilon(j, k, 3) * Pauli(3, jlab)
        return super().__mul__(other)

    def _eval_power(b, e):
        if e.is_Integer and e.is_positive:
            return super().__pow__(int(e) % 2)