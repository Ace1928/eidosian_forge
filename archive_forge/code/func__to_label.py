from __future__ import annotations
import copy
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
@staticmethod
def _to_label(z, x, phase, group_phase=False, full_group=True, return_phase=False):
    """Return the label string for a Pauli.

        Args:
            z (array): The symplectic representation z vector.
            x (array): The symplectic representation x vector.
            phase (int): Pauli phase.
            group_phase (bool): Optional. If ``True`` use group-phase convention
                                instead of BasePauli ZX-phase convention.
                                (default: ``False``).
            full_group (bool): If True return the Pauli label from the full Pauli group
                including complex coefficient from [1, -1, 1j, -1j]. If
                ``False`` return the unsigned Pauli label with coefficient 1
                (default: ``True``).
            return_phase (bool): If ``True`` return the adjusted phase for the coefficient
                of the returned Pauli label. This can be used even if
                ``full_group=False``.

        Returns:
            str: the Pauli label from the full Pauli group (if ``full_group=True``) or
                from the unsigned Pauli group (if ``full_group=False``).
            tuple[str, int]: if ``return_phase=True`` returns a tuple of the Pauli
                            label (from either the full or unsigned Pauli group) and
                            the phase ``q`` for the coefficient :math:`(-i)^(q + x.z)`
                            for the label from the full Pauli group.
        """
    num_qubits = z.size
    phase = int(phase)
    coeff_labels = {0: '', 1: '-i', 2: '-', 3: 'i'}
    label = ''
    for i in range(num_qubits):
        if not z[num_qubits - 1 - i]:
            if not x[num_qubits - 1 - i]:
                label += 'I'
            else:
                label += 'X'
        elif not x[num_qubits - 1 - i]:
            label += 'Z'
        else:
            label += 'Y'
            if not group_phase:
                phase -= 1
    phase %= 4
    if phase and full_group:
        label = coeff_labels[phase] + label
    if return_phase:
        return (label, phase)
    return label