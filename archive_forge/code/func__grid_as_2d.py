from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
import numpy as np
def _grid_as_2d(self, x: ArrayLike, y: ArrayLike) -> tuple[CoordinateArray, CoordinateArray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x, y = np.meshgrid(x, y)
    return (x, y)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x, y = np.meshgrid(x, y)
    return (x, y)