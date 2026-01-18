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
class MultiControlledX(ControlledOp):
    """MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on
        control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates


    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """
    is_self_inverse = True
    'bool: Whether or not the operator is self-inverse.'
    num_wires = AnyWires
    'int: Number of wires the operation acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = ()
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    name = 'MultiControlledX'

    def _flatten(self):
        return ((), (self.active_wires, tuple(self.control_values), self.work_wires))

    @classmethod
    def _unflatten(cls, _, metadata):
        return cls(wires=metadata[0], control_values=metadata[1], work_wires=metadata[2])

    def __init__(self, control_wires=None, wires=None, control_values=None, work_wires=None):
        if wires is None:
            raise ValueError('Must specify the wires where the operation acts on')
        wires = wires if isinstance(wires, Wires) else Wires(wires)
        if control_wires is not None:
            warnings.warn('The control_wires keyword will be removed soon. Use wires = (control_wires, target_wire) instead. See the documentation for more information.', UserWarning)
            if len(wires) != 1:
                raise ValueError('MultiControlledX accepts a single target wire.')
        else:
            if len(wires) < 2:
                raise ValueError(f'MultiControlledX: wrong number of wires. {len(wires)} wire(s) given. Need at least 2.')
            control_wires = wires[:-1]
            wires = wires[-1:]
        control_values = _check_and_convert_control_values(control_values, control_wires)
        super().__init__(qml.X(wires), control_wires=control_wires, control_values=control_values, work_wires=work_wires)

    def __repr__(self):
        return f'MultiControlledX(wires={self.active_wires.tolist()}, control_values={self.control_values})'

    @property
    def wires(self):
        return self.active_wires

    @staticmethod
    def compute_matrix(control_wires, control_values=None, **kwargs):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiControlledX.matrix`

        Args:
            control_wires (Any or Iterable[Any]): wires to place controls on
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            tensor_like: matrix representation

        **Example**

        >>> print(qml.MultiControlledX.compute_matrix([0], [1]))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
        >>> print(qml.MultiControlledX.compute_matrix([1], [0]))
        [[0. 1. 0. 0.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        """
        control_values = _check_and_convert_control_values(control_values, control_wires)
        padding_left = sum((2 ** i * int(val) for i, val in enumerate(reversed(control_values)))) * 2
        padding_right = 2 ** (len(control_wires) + 1) - 2 - padding_left
        return block_diag(np.eye(padding_left), qml.X.compute_matrix(), np.eye(padding_right))

    def matrix(self, wire_order=None):
        canonical_matrix = self.compute_matrix(self.control_wires, self.control_values)
        wire_order = wire_order or self.wires
        return qml.math.expand_matrix(canonical_matrix, wires=self.active_wires, wire_order=wire_order)

    @staticmethod
    def compute_decomposition(wires=None, work_wires=None, control_values=None, **kwargs):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.MultiControlledX.decomposition`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operation acts on
            work_wires (Wires): optional work wires used to decompose
                the operation into a series of Toffoli gates.
            control_values (Union[bool, list[bool], int, list[int]]): The value(s) the control wire(s)
                should take. Integers other than 0 or 1 will be treated as ``int(bool(x))``.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.MultiControlledX.compute_decomposition(
        ...     wires=[0,1,2,3], control_values=[1,1,1], work_wires=qml.wires.Wires("aux")))
        [Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux']),
        Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux'])]

        """
        if len(wires) < 2:
            raise ValueError(f'Wrong number of wires. {len(wires)} given. Need at least 2.')
        target_wire = wires[-1]
        control_wires = wires[:-1]
        if control_values is None:
            control_values = [True] * len(control_wires)
        work_wires = work_wires or []
        if len(control_wires) > 2 and len(work_wires) == 0:
            raise ValueError('At least one work wire is required to decompose operation: MultiControlledX')
        flips1 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]
        if len(control_wires) == 1:
            decomp = [qml.CNOT(wires=wires)]
        elif len(control_wires) == 2:
            decomp = qml.Toffoli.compute_decomposition(wires=wires)
        else:
            decomp = decompose_mcx(control_wires, target_wire, work_wires)
        flips2 = [qml.X(w) for w, val in zip(control_wires, control_values) if not val]
        return flips1 + decomp + flips2

    def decomposition(self):
        return self.compute_decomposition(self.active_wires, self.work_wires, self.control_values)