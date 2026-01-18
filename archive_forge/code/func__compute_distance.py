from __future__ import annotations
import logging
from typing import (
from typing_extensions import TypedDict
def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> np.floating:
    if self.distance == 'cosine':
        return self._cosine_distance(a, b)
    elif self.distance == 'euclidean':
        return self._euclidean_distance(a, b)
    elif self.distance == 'manhattan':
        return self._manhattan_distance(a, b)
    elif self.distance == 'chebyshev':
        return self._chebyshev_distance(a, b)
    elif self.distance == 'hamming':
        return self._hamming_distance(a, b)
    else:
        raise ValueError(f'Invalid distance metric: {self.distance}')