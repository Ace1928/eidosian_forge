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
class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        return (type(self), self.function)

    def __call__(self, *args):
        return self.function(*args)