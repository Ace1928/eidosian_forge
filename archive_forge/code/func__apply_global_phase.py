import functools
import itertools
from string import ascii_letters as ABC
from typing import List
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import BasisState, DeviceError, QubitDevice, StatePrep, Snapshot
from pennylane.devices.qubit import measure
from pennylane.operation import Operation
from pennylane.ops import Sum
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.pulse import ParametrizedEvolution
from pennylane.measurements import ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import WireError
from .._version import __version__
def _apply_global_phase(self, state, operation: qml.GlobalPhase):
    """Applies a :class:`~.GlobalPhase` operation to the state."""
    return qml.math.exp(-1j * operation.data[0]) * state