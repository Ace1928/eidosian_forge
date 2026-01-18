from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _decompose_xx_yy_into_two_fsims_ignoring_single_qubit_ops(*, qubits: Sequence['cirq.Qid'], fsim_gate: 'cirq.FSimGate', canonical_x_kak_coefficient: float, canonical_y_kak_coefficient: float, atol: float=1e-08) -> List['cirq.Operation']:
    x = canonical_x_kak_coefficient
    y = canonical_y_kak_coefficient
    assert 0 <= y <= x <= np.pi / 4
    eta = np.sin(x) ** 2 * np.cos(y) ** 2 + np.cos(x) ** 2 * np.sin(y) ** 2
    xi = abs(np.sin(2 * x) * np.sin(2 * y))
    t = fsim_gate.phi / 2
    assert isinstance(fsim_gate.theta, float)
    kappa = np.sin(fsim_gate.theta) ** 2 - np.sin(t) ** 2
    s_sum = (eta - np.sin(t) ** 2) / kappa
    s_dif = 0.5 * xi / kappa
    a_dif = _sticky_0_to_1(s_sum + s_dif, atol=atol)
    a_sum = _sticky_0_to_1(s_sum - s_dif, atol=atol)
    if a_dif is None or a_sum is None:
        raise ValueError(f'Failed to synthesize XX^{x / np.pi}·YY^{y / np.pi} from two {fsim_gate!r} separated by single qubit operations.')
    x_dif = np.arcsin(np.sqrt(a_dif))
    x_sum = np.arcsin(np.sqrt(a_sum))
    x_a = x_sum + x_dif
    x_b = x_dif - x_sum
    a, b = qubits
    return [fsim_gate(a, b), ops.rz(t + np.pi).on(a), ops.rz(t).on(b), ops.rx(x_a).on(a), ops.rx(x_b).on(b), fsim_gate(a, b)]