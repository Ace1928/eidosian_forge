import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _random_single_q_clifford(qubit: 'cirq.Qid', num_cfds: int, cfds: Sequence[Sequence['cirq.Gate']], cfd_matrices: np.ndarray) -> 'cirq.Circuit':
    clifford_group_size = 24
    gate_ids = list(np.random.choice(clifford_group_size, num_cfds))
    gate_sequence = [gate for gate_id in gate_ids for gate in cfds[gate_id]]
    idx = _find_inv_matrix(_gate_seq_to_mats(gate_sequence), cfd_matrices)
    gate_sequence.extend(cfds[idx])
    circuit = circuits.Circuit((gate(qubit) for gate in gate_sequence))
    return circuit