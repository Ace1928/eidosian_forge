import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
class PowOpObs(PowOperation, Observable):
    """A child class of ``Pow`` that inherits from both
    ``Observable`` and ``Operation``.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)