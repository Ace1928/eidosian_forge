from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
@dataclass
class QuantumVolumeResult:
    """Stores one run of the results and test information used when running the
    quantum volume benchmark so it may be analyzed in detail afterwards.

    """
    model_circuit: cirq.Circuit
    heavy_set: List[int]
    compiled_circuit: cirq.Circuit
    sampler_result: float

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['model_circuit', 'heavy_set', 'compiled_circuit', 'sampler_result'])