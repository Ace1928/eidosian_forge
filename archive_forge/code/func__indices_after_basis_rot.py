import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _indices_after_basis_rot(i: int, j: int) -> Tuple[int, Sequence[int], Sequence[int]]:
    mat_idx = 3 * (3 * i + j)
    q_0_i = 3 - i
    q_1_j = 3 - j
    indices = [q_1_j - 1, 4 * q_0_i - 1, 4 * q_0_i + q_1_j - 1]
    signs = [(-1) ** (j == 2), (-1) ** (i == 2), (-1) ** ((i == 2) + (j == 2))]
    return (mat_idx, indices, signs)