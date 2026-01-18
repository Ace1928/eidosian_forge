import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
class ControlledQubitUnitary(ControlledOp):
    """ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``control_values``: the state on which to apply the controlled operation (see below)
    * ``target_wires``: the wires the unitary matrix will be applied to
    * ``active_wires``: Wires modified by the operator. This is the control wires followed
        by the target wires.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        base (Union[array[complex], QubitUnitary]): square unitary matrix or a QubitUnitary
            operation. If passing a matrix, this will be used to construct a QubitUnitary
            operator that will be used as the base operator. If providing a ``qml.QubitUnitary``,
            this will be used as the base directly.
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
            (optional if U is provided as a QubitUnitary)
        control_values (List[int, bool]): a list providing the state of the control qubits to
            control on (default is the all 1s state)
        unitary_check (bool): whether to check whether an array U is unitary when creating the
            operator (default False)

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Alternatively, the same operator can be constructed with a QubitUnitary:

    >>> base = qml.QubitUnitary(U, wires=2)
    >>> qml.ControlledQubitUnitary(base, control_wires=[0, 1])
    Controlled(QubitUnitary(array([[ 0.94877869,  0.31594146],
       [-0.31594146,  0.94877869]]), wires=[2]), control_wires=[0, 1])

    Typically, controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\\vert 1\\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\\vert 0\\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[0, 1, 1])

    or

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values=[False, True, True])
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], control_wires=metadata[0], control_values=metadata[1], work_wires=metadata[2])

    def __init__(self, base, control_wires, wires=None, control_values=None, unitary_check=False, work_wires=None):
        if getattr(base, 'wires', False) and wires is not None:
            warnings.warn('base operator already has wires; values specified through wires kwarg will be ignored.')
        if isinstance(base, Iterable):
            base = QubitUnitary(base, wires=wires, unitary_check=unitary_check)
        super().__init__(base, control_wires, control_values=control_values, work_wires=work_wires)
        self._name = 'ControlledQubitUnitary'

    def _controlled(self, wire):
        ctrl_wires = wire + self.control_wires
        values = None if self.control_values is None else [True] + self.control_values
        return ControlledQubitUnitary(self.base, control_wires=ctrl_wires, control_values=values, work_wires=self.work_wires)

    @property
    def has_decomposition(self):
        if not super().has_decomposition:
            return False
        with qml.QueuingManager.stop_recording():
            try:
                self.decomposition()
            except qml.operation.DecompositionUndefinedError:
                return False
        return True