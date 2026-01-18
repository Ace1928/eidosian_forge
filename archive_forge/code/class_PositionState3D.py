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
class PositionState3D(State):
    """ Base class for 3D cartesian position eigenstates """

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class, *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ('x', 'y', 'z')

    @property
    def position_x(self):
        """ The x coordinate of the state """
        return self.label[0]

    @property
    def position_y(self):
        """ The y coordinate of the state """
        return self.label[1]

    @property
    def position_z(self):
        """ The z coordinate of the state """
        return self.label[2]