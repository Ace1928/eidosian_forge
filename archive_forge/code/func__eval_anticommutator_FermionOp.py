from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.physics.quantum import Operator
from sympy.physics.quantum import HilbertSpace, Ket, Bra
from sympy.functions.special.tensor_functions import KroneckerDelta
def _eval_anticommutator_FermionOp(self, other, **hints):
    if self.name == other.name:
        if not self.is_annihilation and other.is_annihilation:
            return S.One
    elif 'independent' in hints and hints['independent']:
        return 2 * self * other
    return None