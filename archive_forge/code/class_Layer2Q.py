from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
class Layer2Q(LayerBase):
    """
    Layer represents a simple circuit where 2-qubit gate matrix (of size 4x4)
    interleaves with the identity ones.
    """

    def __init__(self, num_qubits: int, j: int, k: int, g4x4: Optional[np.ndarray]=None):
        """
        Args:
            num_qubits: number of qubits.
            j: index of the first (control) bit.
            k: index of the second (target) bit.
            g4x4: 4x4 matrix that makes up this layer along with identity ones,
                  or None (should be set up later).
        """
        super().__init__()
        self._gmat = np.full((4, 4), fill_value=0, dtype=np.complex128)
        if isinstance(g4x4, np.ndarray):
            np.copyto(self._gmat, g4x4)
        bit_flip = True
        dim = 2 ** num_qubits
        row_perm = reverse_bits(bit_permutation_2q(n=num_qubits, j=j, k=k), nbits=num_qubits, enable=bit_flip)
        col_perm = reverse_bits(np.arange(dim, dtype=np.int64), nbits=num_qubits, enable=bit_flip)
        self._perm = np.full((dim,), fill_value=0, dtype=np.int64)
        self._perm[row_perm] = col_perm
        self._inv_perm = inverse_permutation(self._perm)

    def set_from_matrix(self, mat: np.ndarray):
        """See base class description."""
        np.copyto(self._gmat, mat)

    def get_attr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See base class description."""
        return (self._gmat, self._perm, self._inv_perm)