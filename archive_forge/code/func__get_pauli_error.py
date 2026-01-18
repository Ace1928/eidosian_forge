import abc
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List, Set, Type
from cirq import _compat, ops, devices
from cirq.devices import noise_utils
def _get_pauli_error(self, p_error: float, op_id: noise_utils.OpIdentifier):
    time_ns = float(self.gate_times_ns[op_id.gate_type])
    for q in op_id.qubits:
        p_error -= noise_utils.decoherence_pauli_error(self.t1_ns[q], self.tphi_ns[q], time_ns)
    return p_error