import cmath
import math
from typing import (
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq._import import LazyLoader
from cirq.linalg import combinators, diagonalize, predicates, transformations
def _rotation_matrix(angle: float) -> np.ndarray:
    c, s = (np.cos(angle), np.sin(angle))
    return np.array([[c, -s], [s, c]])