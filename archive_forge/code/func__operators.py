import copy
import numpy as np
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation
@property
def _operators(self) -> list[qml.operation.Operator]:
    """Flattened list of operators that compose this QSVT operation."""
    return [self._hyperparameters['UA'], *self._hyperparameters['projectors']]