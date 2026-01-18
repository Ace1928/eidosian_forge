from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
@dataclass
class XXPolytope:
    """
    Describes those two-qubit programs accessible to a given sequence of XX-type interactions.

    NOTE: Strengths are normalized so that CX corresponds to pi / 4, which differs from Qiskit's
          conventions around RZX elsewhere.
    """
    total_strength: float = 0.0
    max_strength: float = 0.0
    place_strength: float = 0.0

    @classmethod
    def from_strengths(cls, *strengths):
        """
        Constructs an XXPolytope from a sequence of strengths.
        """
        total_strength, max_strength, place_strength = (0, 0, 0)
        for strength in strengths:
            total_strength += strength
            if strength >= max_strength:
                max_strength, place_strength = (strength, max_strength)
            elif strength >= place_strength:
                place_strength = strength
        return XXPolytope(total_strength=total_strength, max_strength=max_strength, place_strength=place_strength)

    def add_strength(self, new_strength: float=0.0):
        """
        Returns a new XXPolytope with one new XX interaction appended.
        """
        return XXPolytope(total_strength=self.total_strength + new_strength, max_strength=max(self.max_strength, new_strength), place_strength=self.max_strength if new_strength > self.max_strength else new_strength if new_strength > self.place_strength else self.place_strength)

    @property
    def _offsets(self):
        """
        Returns b with A*x + b ≥ 0 iff x belongs to the XXPolytope.
        """
        return np.array([0, 0, 0, np.pi / 2, self.total_strength, self.total_strength - 2 * self.max_strength, self.total_strength - self.max_strength - self.place_strength])

    def member(self, point):
        """
        Returns True when `point` is a member of `self`.
        """
        reflected_point = point.copy().reshape(-1, 3)
        rows = reflected_point[:, 0] >= np.pi / 4 + EPSILON
        reflected_point[rows, 0] = np.pi / 2 - reflected_point[rows, 0]
        reflected_point = reflected_point.reshape(point.shape)
        return np.all(self._offsets + np.einsum('ij,...j->...i', A, reflected_point) >= -EPSILON, axis=-1)

    def nearest(self, point):
        """
        Finds the nearest point (in Euclidean or infidelity distance) to `self`.
        """
        if isinstance(point, np.ndarray) and len(point.shape) == 1:
            y0 = point.copy()
        elif isinstance(point, list):
            y0 = np.array(point)
        else:
            raise TypeError(f"Can't handle type of point: {point} ({type(point)})")
        reflected_p = y0[0] > np.pi / 4 + EPSILON
        if reflected_p:
            y0[0] = np.pi / 2 - y0[0]
        if self.member(y0):
            if reflected_p:
                y0[0] = np.pi / 2 - y0[0]
            return y0
        b1 = self._offsets.reshape(7, 1)
        A1y0 = np.einsum('ijk,k->ij', A1, y0)
        nearest1 = np.einsum('ijk,ik->ij', A1inv, b1 + A1y0) - y0
        b2 = np.array([*combinations(self._offsets, 2)])
        A2y0 = np.einsum('ijk,k->ij', A2, y0)
        nearest2 = np.einsum('ijk,ik->ij', A2inv, b2 + A2y0) - y0
        b3 = np.array([*combinations(self._offsets, 3)])
        nearest3 = np.einsum('ijk,ik->ij', A3inv, b3)
        nearest = -np.concatenate([nearest1, nearest2, nearest3])
        nearest = nearest[self.member(nearest)]
        smallest_index = np.argmin(np.linalg.norm(nearest - y0, axis=1))
        if reflected_p:
            nearest[smallest_index][0] = np.pi / 2 - nearest[smallest_index][0]
        return nearest[smallest_index]