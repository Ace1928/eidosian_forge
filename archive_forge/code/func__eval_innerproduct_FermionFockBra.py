from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
def _eval_innerproduct_FermionFockBra(self, bra, **hints):
    return KroneckerDelta(self.n, bra.n)