import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
def _get_idx_to_keep(self, qid_map: Mapping[raw_types.Qid, int]):
    num_qubits = len(qid_map)
    idx_to_keep: List[Any] = [slice(0, 2)] * num_qubits
    for q in self.projector_dict.keys():
        idx_to_keep[qid_map[q]] = self.projector_dict[q]
    return tuple(idx_to_keep)