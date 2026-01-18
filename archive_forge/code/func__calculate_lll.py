from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def _calculate_lll(self, delta: float=0.75) -> tuple[np.ndarray, np.ndarray]:
    """Performs a Lenstra-Lenstra-Lovasz lattice basis reduction to obtain a
        c-reduced basis. This method returns a basis which is as "good" as
        possible, with "good" defined by orthogonality of the lattice vectors.

        This basis is used for all the periodic boundary condition calculations.

        Args:
            delta (float): Reduction parameter. Default of 0.75 is usually fine.

        Returns:
            Reduced lattice matrix, mapping to get to that lattice.
        """
    a = self._matrix.copy().T
    b = np.zeros((3, 3))
    u = np.zeros((3, 3))
    m = np.zeros(3)
    b[:, 0] = a[:, 0]
    m[0] = np.dot(b[:, 0], b[:, 0])
    for i in range(1, 3):
        u[i, 0:i] = np.dot(a[:, i].T, b[:, 0:i]) / m[0:i]
        b[:, i] = a[:, i] - np.dot(b[:, 0:i], u[i, 0:i].T)
        m[i] = np.dot(b[:, i], b[:, i])
    k = 2
    mapping = np.identity(3, dtype=np.double)
    while k <= 3:
        for i in range(k - 1, 0, -1):
            q = round(u[k - 1, i - 1])
            if q != 0:
                a[:, k - 1] = a[:, k - 1] - q * a[:, i - 1]
                mapping[:, k - 1] = mapping[:, k - 1] - q * mapping[:, i - 1]
                uu = list(u[i - 1, 0:i - 1])
                uu.append(1)
                u[k - 1, 0:i] = u[k - 1, 0:i] - q * np.array(uu)
        if np.dot(b[:, k - 1], b[:, k - 1]) >= (delta - abs(u[k - 1, k - 2]) ** 2) * np.dot(b[:, k - 2], b[:, k - 2]):
            k += 1
        else:
            v = a[:, k - 1].copy()
            a[:, k - 1] = a[:, k - 2].copy()
            a[:, k - 2] = v
            v_m = mapping[:, k - 1].copy()
            mapping[:, k - 1] = mapping[:, k - 2].copy()
            mapping[:, k - 2] = v_m
            for s in range(k - 1, k + 1):
                u[s - 1, 0:s - 1] = np.dot(a[:, s - 1].T, b[:, 0:s - 1]) / m[0:s - 1]
                b[:, s - 1] = a[:, s - 1] - np.dot(b[:, 0:s - 1], u[s - 1, 0:s - 1].T)
                m[s - 1] = np.dot(b[:, s - 1], b[:, s - 1])
            if k > 2:
                k -= 1
            else:
                p = np.dot(a[:, k:3].T, b[:, k - 2:k])
                q = np.diag(m[k - 2:k])
                result = np.linalg.lstsq(q.T, p.T, rcond=None)[0].T
                u[k:3, k - 2:k] = result
    return (a.T, mapping.T)