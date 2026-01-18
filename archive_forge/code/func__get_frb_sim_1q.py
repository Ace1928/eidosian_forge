from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def _get_frb_sim_1q(node_id: int, benchmarks: List[Operation]) -> Optional[float]:
    frb_sim_1q = next((benchmark for benchmark in benchmarks if benchmark.name == 'randomized_benchmark_simultaneous_1q'), None)
    if frb_sim_1q is None:
        return None
    site = next((characteristic for characteristic in frb_sim_1q.sites[0].characteristics if isinstance(characteristic.node_ids, list) and len(characteristic.node_ids) == 1 and (characteristic.node_ids[0] == node_id)), None)
    if site is None:
        return None
    return site.value