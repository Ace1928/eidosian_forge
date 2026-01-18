from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Optional
import numpy as np
from .fast_grad_utils import (
class LayerBase(ABC):
    """
    Base class for any layer implementation. Each layer here is represented
    by a 2x2 or 4x4 gate matrix ``G`` (applied to 1 or 2 qubits respectively)
    interleaved with the identity ones:
    ``Layer = I kron I kron ... kron G kron ... kron I kron I``
    """

    @abstractmethod
    def set_from_matrix(self, mat: np.ndarray):
        """
        Updates this layer from an external gate matrix.

        Args:
            mat: external gate matrix that initializes this layer's one.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_attr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns gate matrix, direct and inverse permutations.

        Returns:
            (1) gate matrix; (2) direct permutation; (3) inverse permutations.
        """
        raise NotImplementedError()