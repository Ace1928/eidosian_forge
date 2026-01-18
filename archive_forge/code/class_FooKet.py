from sympy.core.numbers import (I, Integer)
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.state import Bra, Ket, StateBase
class FooKet(Ket, FooState):

    @classmethod
    def dual_class(self):
        return FooBra

    def _eval_innerproduct_FooBra(self, bra):
        return Integer(1)

    def _eval_innerproduct_BarBra(self, bra):
        return I