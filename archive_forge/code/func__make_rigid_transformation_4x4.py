import collections
import copy
import functools
from importlib import resources
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
def _make_rigid_transformation_4x4(ex: np.ndarray, ey: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a rigid 4x4 transformation matrix from two axes and transl."""
    ex_normalized = ex / np.linalg.norm(ex)
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m