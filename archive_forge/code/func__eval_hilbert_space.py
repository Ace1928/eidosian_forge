from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit
@classmethod
def _eval_hilbert_space(cls, args):
    """This returns the smallest possible Hilbert space."""
    return ComplexSpace(2) ** args[0]