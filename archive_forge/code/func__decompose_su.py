from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _decompose_su(matrix: np.ndarray, controls: List['cirq.Qid'], target: 'cirq.Qid') -> List['cirq.Operation']:
    """Decomposes controlled special unitary gate into elementary gates.

    Result has O(len(controls)) operations.
    See [1], lemma 7.9.
    """
    assert matrix.shape == (2, 2)
    assert is_special_unitary(matrix)
    assert len(controls) >= 1
    a, b, c, _ = _decompose_abc(matrix)
    cnots = decompose_multi_controlled_x(controls[:-1], target, [controls[-1]])
    return [*_decompose_single_ctrl(c, controls[-1], target), *cnots, *_decompose_single_ctrl(b, controls[-1], target), *cnots, *_decompose_single_ctrl(a, controls[-1], target)]