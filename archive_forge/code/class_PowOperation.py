import copy
from typing import Union
from scipy.linalg import fractional_matrix_power
import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply
from .symbolicop import ScalarSymbolicOp
class PowOperation(Pow, Operation):
    """Operation-specific methods and properties for the ``Pow`` class.

    Dynamically mixed in based on the provided base operator.  If the base operator is an
    Operation, this class will be mixed in.

    When we no longer rely on certain functionality through `Operation`, we can get rid of this
    class.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)
    grad_method = None

    @property
    def name(self):
        return self._name

    @property
    def control_wires(self):
        return self.base.control_wires