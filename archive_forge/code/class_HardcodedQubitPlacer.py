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
class HardcodedQubitPlacer(QubitPlacer):

    def __init__(self, mapping: Dict[cirq.NamedTopology, Dict[Any, cirq.Qid]], topo_node_to_qubit_func: Callable[[Hashable], cirq.Qid]=default_topo_node_to_qubit):
        """A placement strategy that uses the explicitly provided `mapping`.

        Args:
            mapping: The hardcoded placements. This provides a placement for each supported
                `cirq.NamedTopology`. The topology serves as the key for the mapping dictionary.
                Each placement is a dictionary mapping topology node to final `cirq.Qid` device
                qubit.
            topo_node_to_qubit_func: A function that maps from `cirq.NamedTopology` nodes
                to `cirq.Qid`. There is a correspondence between nodes and the "abstract" Qids
                used to construct the un-placed circuit. We use this function to interpret
                the provided mappings. By default: nodes which are tuples correspond
                to `cirq.GridQubit`s; otherwise `cirq.LineQubit`.

        Note:
            The attribute `topo_node_to_qubit_func` is not preserved in JSON serialization. This
            bit of plumbing does not affect the placement behavior.
        """
        self._mapping = mapping
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(self, circuit: cirq.AbstractCircuit, problem_topology: NamedTopology, shared_rt_info: 'cg.SharedRuntimeInfo', rs: np.random.RandomState) -> Tuple[cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        """Place a circuit according to the hardcoded placements.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit, use to look
                up the placement in `self.mapping`.
            shared_rt_info: A `cg.SharedRuntimeInfo` object; ignored for hardcoded placement.
            rs: A `RandomState`; ignored for hardcoded placement.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.

        Raises:
            CouldNotPlaceError: if the given problem_topology is not present in the hardcoded
                mapping.
        """
        try:
            nt_mapping = self._mapping[problem_topology]
        except KeyError as e:
            raise CouldNotPlaceError(str(e))
        circuit_mapping = {self.topo_node_to_qubit_func(nt_node): gridq for nt_node, gridq in nt_mapping.items()}
        circuit = circuit.unfreeze().transform_qubits(circuit_mapping).freeze()
        return (circuit, circuit_mapping)

    def __repr__(self) -> str:
        return f'cirq_google.HardcodedQubitPlacer(mapping={_compat.proper_repr(self._mapping)})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        d = obj_to_dict_helper(self, attribute_names=[])
        mapping = {topo: list(placement.items()) for topo, placement in self._mapping.items()}
        mapping = list(mapping.items())
        d['mapping'] = mapping
        return d

    @classmethod
    def _from_json_dict_(cls, **kwargs) -> 'HardcodedQubitPlacer':
        mapping: Dict[cirq.NamedTopology, Dict[Any, 'cirq.Qid']] = {}
        for topo, placement_kvs in kwargs['mapping']:
            placement: Dict[Hashable, 'cirq.Qid'] = {}
            for k, v in placement_kvs:
                if isinstance(k, list):
                    k = tuple(k)
                placement[k] = v
            mapping[topo] = placement
        return cls(mapping=mapping)

    def __eq__(self, other):
        if not isinstance(other, HardcodedQubitPlacer):
            return False
        return self._mapping == other._mapping