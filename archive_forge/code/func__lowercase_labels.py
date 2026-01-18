from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import L2
from sympy.physics.quantum.operator import DifferentialOperator, HermitianOperator
from sympy.physics.quantum.state import Ket, Bra, State
def _lowercase_labels(ops):
    if not isinstance(ops, set):
        ops = [ops]
    return [str(arg.label[0]).lower() for arg in ops]