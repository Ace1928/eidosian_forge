import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _split_two_q_clifford_idx(idx: int):
    """Decompose the index for two-qubit Cliffords."""
    idx_0 = int(idx / 480)
    idx_1 = int(idx % 480 * 0.05)
    idx_2 = idx - idx_0 * 480 - idx_1 * 20
    return (idx_0, idx_1, idx_2)