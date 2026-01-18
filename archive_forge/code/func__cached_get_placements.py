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
@lru_cache()
def _cached_get_placements(problem_topo: 'cirq.NamedTopology', device: 'cirq.Device') -> List[Dict[Any, 'cirq.Qid']]:
    """Cache `cirq.get_placements` onto the specific device."""
    return get_placements(big_graph=_Device_dot_get_nx_graph(device), small_graph=problem_topo.graph)