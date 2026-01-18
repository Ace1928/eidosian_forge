import copy
import functools
from warnings import warn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Tuple, Optional, Union
import pennylane as qml
from pennylane.operation import Operator, DecompositionUndefinedError, EigvalsUndefinedError
from pennylane.pytrees import register_pytree
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .shots import Shots
from a classical shadow measurement"""
class StateMeasurement(MeasurementProcess):
    """State-based measurement process.

    Any class inheriting from ``StateMeasurement`` should define its own ``process_state`` method,
    which should have the following arguments:

    * state (Sequence[complex]): quantum state with a flat shape. It may also have an
        optional batch dimension
    * wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
        dimension :math:`2^n` acts on a subspace of :math:`n` wires

    **Example:**

    Let's create a measurement that returns the diagonal of the reduced density matrix.

    >>> class MyMeasurement(StateMeasurement):
    ...     def process_state(self, state, wire_order):
    ...         # use the already defined `qml.density_matrix` measurement to compute the
    ...         # reduced density matrix from the given state
    ...         density_matrix = qml.density_matrix(wires=self.wires).process_state(state, wire_order)
    ...         return qml.math.diagonal(qml.math.real(density_matrix))

    We can now execute it in a QNode:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.Hadamard(0)
    ...     qml.CNOT([0, 1])
    ...     return MyMeasurement(wires=[0])
    >>> circuit()
    tensor([0.5, 0.5], requires_grad=True)
    """

    @abstractmethod
    def process_state(self, state: Sequence[complex], wire_order: Wires):
        """Process the given quantum state.

        Args:
            state (Sequence[complex]): quantum state with a flat shape. It may also have an
                optional batch dimension
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
        """