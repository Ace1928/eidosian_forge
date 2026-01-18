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
class RandomDevicePlacer(QubitPlacer):

    def __init__(self, topo_node_to_qubit_func: Callable[[Any], cirq.Qid]=default_topo_node_to_qubit):
        """A placement strategy that randomly places circuits onto devices.

        Args:
            topo_node_to_qubit_func: A function that maps from `cirq.NamedTopology` nodes
                to `cirq.Qid`. There is a correspondence between nodes and the "abstract" Qids
                used to construct the un-placed circuit. `cirq.get_placements` returns a dictionary
                mapping from node to Qid. We use this function to transform it into a mapping
                from "abstract" Qid to device Qid. By default: nodes which are tuples correspond
                to `cirq.GridQubit`s; otherwise `cirq.LineQubit`.

        Note:
            The attribute `topo_node_to_qubit_func` is not preserved in JSON serialization. This
            bit of plumbing does not affect the placement behavior.
        """
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(self, circuit: 'cirq.AbstractCircuit', problem_topology: 'cirq.NamedTopology', shared_rt_info: 'cg.SharedRuntimeInfo', rs: np.random.RandomState) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        """Place a circuit with a given topology onto a device via `cirq.get_placements` with
        randomized selection of the placement each time.

        This requires device information to be present in `shared_rt_info`.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit.
            shared_rt_info: A `cg.SharedRuntimeInfo` object that contains a `device` attribute
                of type `cirq.Device` to enable placement.
            rs: A `RandomState` as a source of randomness for random placements.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.

        Raises:
            ValueError: If `shared_rt_info` does not have a device field.
        """
        device = shared_rt_info.device
        if device is None:
            raise ValueError('RandomDevicePlacer requires shared_rt_info.device to be a `cirq.Device`. This should have been set during the initialization phase of `cg.execute`.')
        placement = _get_random_placement(problem_topology, device, rs=rs, topo_node_to_qubit_func=self.topo_node_to_qubit_func)
        return (circuit.unfreeze().transform_qubits(placement).freeze(), placement)

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __repr__(self) -> str:
        return 'cirq_google.RandomDevicePlacer()'

    def __eq__(self, other):
        if isinstance(other, RandomDevicePlacer):
            return True