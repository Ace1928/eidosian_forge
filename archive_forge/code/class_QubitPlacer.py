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
class QubitPlacer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def place_circuit(self, circuit: cirq.AbstractCircuit, problem_topology: 'cirq.NamedTopology', shared_rt_info: 'cg.SharedRuntimeInfo', rs: np.random.RandomState) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        """Place a circuit with a given topology.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit.
            shared_rt_info: A `cg.SharedRuntimeInfo` object that may contain additional info
                to inform placement.
            rs: A `RandomState` to enable pseudo-random placement strategies.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.
        """