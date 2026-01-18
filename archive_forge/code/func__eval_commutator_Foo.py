from sympy.core.numbers import Integer
from sympy.core.symbol import symbols
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.operator import Operator
def _eval_commutator_Foo(self, foo):
    return Integer(1)