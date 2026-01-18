import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _two_qubit_clifford_mixers(q_0: 'cirq.Qid', q_1: 'cirq.Qid', idx_2: int, cliffords: Cliffords) -> Iterator['cirq.OP_TREE']:
    """Fulfills parts (b-d) for two-qubit Cliffords."""
    s1 = cliffords.s1
    s1_x = cliffords.s1_x
    s1_y = cliffords.s1_y
    if idx_2 == 1:
        yield ops.CZ(q_0, q_1)
        yield (ops.Y(q_0) ** (-0.5))
        yield (ops.Y(q_1) ** 0.5)
        yield ops.CZ(q_0, q_1)
        yield (ops.Y(q_0) ** 0.5)
        yield (ops.Y(q_1) ** (-0.5))
        yield ops.CZ(q_0, q_1)
        yield (ops.Y(q_1) ** 0.5)
    elif 2 <= idx_2 <= 10:
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 2) / 3)
        idx_4 = (idx_2 - 2) % 3
        yield _single_qubit_gates(s1[idx_3], q_0)
        yield _single_qubit_gates(s1_y[idx_4], q_1)
    elif idx_2 >= 11:
        yield ops.CZ(q_0, q_1)
        yield (ops.Y(q_0) ** 0.5)
        yield (ops.X(q_1) ** (-0.5))
        yield ops.CZ(q_0, q_1)
        idx_3 = int((idx_2 - 11) / 3)
        idx_4 = (idx_2 - 11) % 3
        yield _single_qubit_gates(s1_y[idx_3], q_0)
        yield _single_qubit_gates(s1_x[idx_4], q_1)