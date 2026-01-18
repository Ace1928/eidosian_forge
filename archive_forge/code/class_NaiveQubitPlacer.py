import abc
import dataclasses
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Callable, TYPE_CHECKING, Hashable
import numpy as np
import cirq
from cirq import _compat
from cirq.devices.named_topologies import get_placements, NamedTopology
from cirq.protocols import obj_to_dict_helper
from cirq_google.workflow._device_shim import _Device_dot_get_nx_graph
@dataclasses.dataclass(frozen=True)
class NaiveQubitPlacer(QubitPlacer):
    """Don't do any qubit placement, use circuit qubits."""

    def place_circuit(self, circuit: 'cirq.AbstractCircuit', problem_topology: 'cirq.NamedTopology', shared_rt_info: 'cg.SharedRuntimeInfo', rs: np.random.RandomState) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        return (circuit.freeze(), {q: q for q in circuit.all_qubits()})

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')