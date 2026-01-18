from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
Fock state bra for a fermionic mode.

    Parameters
    ==========

    n : Number
        The Fock state number.

    