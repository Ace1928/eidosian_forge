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
def _get_random_placement(problem_topology: 'cirq.NamedTopology', device: 'cirq.Device', rs: np.random.RandomState, topo_node_to_qubit_func: Callable[[Any], 'cirq.Qid']=default_topo_node_to_qubit) -> Dict['cirq.Qid', 'cirq.Qid']:
    """Place `problem_topology` randomly onto a device.

    This is a helper function used by `RandomDevicePlacer.place_circuit`.
    """
    placements = _cached_get_placements(problem_topology, device)
    if len(placements) == 0:
        raise CouldNotPlaceError
    random_i = rs.randint(len(placements))
    placement = placements[random_i]
    placement_gq = {topo_node_to_qubit_func(k): v for k, v in placement.items()}
    return placement_gq