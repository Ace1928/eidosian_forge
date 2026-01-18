import sys
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union, cast
import numpy as np
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import I, RX, MEASURE
from pyquil.noise_gates import _get_qvm_noise_supported_gates
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator
from pyquil.quilbase import Pragma, Gate, Declare
def _decoherence_noise_model(gates: Sequence[Gate], T1: Union[Dict[int, float], float]=3e-05, T2: Union[Dict[int, float], float]=3e-05, gate_time_1q: float=5e-08, gate_time_2q: float=1.5e-07, ro_fidelity: Union[Dict[int, float], float]=0.95) -> NoiseModel:
    """
    The default noise parameters

    - T1 = 30 us
    - T2 = 30 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A NoiseModel with the appropriate Kraus operators defined.
    """
    all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))
    if isinstance(T1, dict):
        all_qubits.update(T1.keys())
    if isinstance(T2, dict):
        all_qubits.update(T2.keys())
    if isinstance(ro_fidelity, dict):
        all_qubits.update(ro_fidelity.keys())
    if not isinstance(T1, dict):
        T1 = {q: T1 for q in all_qubits}
    if not isinstance(T2, dict):
        T2 = {q: T2 for q in all_qubits}
    if not isinstance(ro_fidelity, dict):
        ro_fidelity = {q: ro_fidelity for q in all_qubits}
    noisy_identities_1q = {q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_1q) for q in all_qubits}
    noisy_identities_2q = {q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_2q) for q in all_qubits}
    kraus_maps = []
    for g in gates:
        targets = tuple((t.index for t in g.qubits))
        if g.name in NO_NOISE:
            continue
        matrix, _ = get_noisy_gate(g.name, g.params)
        if len(targets) == 1:
            noisy_I = noisy_identities_1q[targets[0]]
        else:
            if len(targets) != 2:
                raise ValueError('Noisy gates on more than 2Q not currently supported')
            noisy_I = tensor_kraus_maps(noisy_identities_2q[targets[1]], noisy_identities_2q[targets[0]])
        kraus_maps.append(KrausModel(g.name, tuple(g.params), targets, combine_kraus_maps(noisy_I, [matrix]), 1.0))
    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1.0 - f_ro], [1.0 - f_ro, f_ro]])
    return NoiseModel(kraus_maps, aprobs)