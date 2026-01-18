from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class SeparationPlane(AbstractChemenvAlgorithm):
    """
    Class representing the algorithm using separation planes for the calculation of
    the Continuous Symmetry Measure.
    """

    def __init__(self, plane_points, mirror_plane=False, ordered_plane=False, point_groups=None, ordered_point_groups=None, explicit_permutations=None, minimum_number_of_points=None, explicit_optimized_permutations=None, multiplicity=None, other_plane_points=None):
        """Initializes a separation plane for a given perfect coordination geometry.

        Args:
            plane_points: Indices of the points that are in the plane in the perfect structure (and should be
                found in the defective one as well).
            mirror_plane: True if the separation plane is a mirror plane, in which case there is a correspondence
                of the points in each point_group (can reduce the number of permutations).
            ordered_plane: True if the order of the points in the plane can be taken into account to reduce the
                number of permutations.
            point_groups: Indices of the points in the two groups of points separated by the plane.
            ordered_point_groups: Whether the order of the points in each group of points can be taken into account to
                reduce the number of permutations.
            explicit_permutations: Explicit permutations to be performed in this separation plane algorithm.
            minimum_number_of_points: Minimum number of points needed to initialize a separation plane
                for this algorithm.
            explicit_optimized_permutations: Optimized set of explicit permutations to be performed in this
                separation plane algorithm.
            multiplicity: Number of such planes in the model geometry.
            other_plane_points: Indices of the points that are in the plane in the perfect structure for the other
                planes. The multiplicity should be equal to the length of this list + 1 ("main" separation plane +
                the other ones).
        """
        super().__init__(algorithm_type=SEPARATION_PLANE)
        self.mirror_plane = mirror_plane
        self.plane_points = plane_points
        self.point_groups = point_groups
        if len(point_groups[0]) > len(point_groups[1]):
            raise RuntimeError('The number of points in the first group should be\nless than or equal to the number of points in the second group')
        self._hash = 10000 * len(plane_points) + 100 * len(point_groups[0]) + len(point_groups[1])
        self.ordered_plane = ordered_plane
        self.ordered_point_groups = [False, False] if ordered_point_groups is None else ordered_point_groups
        self.explicit_permutations = explicit_permutations
        self.explicit_optimized_permutations = explicit_optimized_permutations
        self._safe_permutations = None
        if self.explicit_optimized_permutations is not None:
            self._permutations = self.explicit_optimized_permutations
        elif self.explicit_permutations is not None:
            self._permutations = self.explicit_permutations
        self.multiplicity = multiplicity
        self.other_plane_points = other_plane_points
        self.minimum_number_of_points = minimum_number_of_points
        self.maximum_number_of_points = len(self.plane_points)
        self._ref_separation_perm = list(self.point_groups[0])
        self._ref_separation_perm.extend(list(self.plane_points))
        self._ref_separation_perm.extend(list(self.point_groups[1]))
        self._argsorted_ref_separation_perm = list(np.argsort(self._ref_separation_perm))
        self.separation = (len(point_groups[0]), len(plane_points), len(point_groups[1]))

    @property
    def permutations(self):
        """
        Permutations used for this separation plane algorithm.

        Returns:
            list[Permutations]: to be performed.
        """
        return self._permutations

    @property
    def ref_separation_perm(self):
        """
        Ordered indices of the separation plane.

        Examples:
            For a separation plane of type 2|4|3, with plane_points indices [0, 3, 5, 8] and
            point_groups indices [1, 4] and [2, 7, 6], the list of ordered indices is :
            [0, 3, 5, 8, 1, 4, 2, 7, 6].

        Returns:
            list[int]: of ordered indices of this separation plane.
        """
        return self._ref_separation_perm

    @property
    def argsorted_ref_separation_perm(self):
        """
        "Arg sorted" ordered indices of the separation plane.

        This is used in the identification of the final permutation to be used.

        Returns:
            list[int]: "arg sorted" ordered indices of the separation plane.
        """
        return self._argsorted_ref_separation_perm

    def safe_separation_permutations(self, ordered_plane=False, ordered_point_groups=None, add_opposite=False):
        """
        Simple and safe permutations for this separation plane.

        This is not meant to be used in production. Default configuration for ChemEnv does not use this method.

        Args:
            ordered_plane: Whether the order of the points in the plane can be used to reduce the
                number of permutations.
            ordered_point_groups: Whether the order of the points in each point group can be used to reduce the
                number of permutations.
            add_opposite: Whether to add the permutations from the second group before the first group as well.

        Returns:
            list[int]: safe permutations.
        """
        s0 = list(range(len(self.point_groups[0])))
        plane = list(range(len(self.point_groups[0]), len(self.point_groups[0]) + len(self.plane_points)))
        s1 = list(range(len(self.point_groups[0]) + len(self.plane_points), len(self.point_groups[0]) + len(self.plane_points) + len(self.point_groups[1])))
        ordered_point_groups = [False, False] if ordered_point_groups is None else ordered_point_groups

        def rotate(s, n):
            return s[-n:] + s[:-n]
        if ordered_plane and self.ordered_plane:
            plane_perms = [rotate(plane, ii) for ii in range(len(plane))]
            inv_plane = plane[::-1]
            plane_perms.extend([rotate(inv_plane, ii) for ii in range(len(inv_plane))])
        else:
            plane_perms = list(itertools.permutations(plane))
        if ordered_point_groups[0] and self.ordered_point_groups[0]:
            s0_perms = [rotate(s0, ii) for ii in range(len(s0))]
            inv_s0 = s0[::-1]
            s0_perms.extend([rotate(inv_s0, ii) for ii in range(len(inv_s0))])
        else:
            s0_perms = list(itertools.permutations(s0))
        if ordered_point_groups[1] and self.ordered_point_groups[1]:
            s1_perms = [rotate(s1, ii) for ii in range(len(s1))]
            inv_s1 = s1[::-1]
            s1_perms.extend([rotate(inv_s1, ii) for ii in range(len(inv_s1))])
        else:
            s1_perms = list(itertools.permutations(s1))
        if self._safe_permutations is None:
            self._safe_permutations = []
            for perm_side1 in s0_perms:
                for perm_sep_plane in plane_perms:
                    for perm_side2 in s1_perms:
                        perm = list(perm_side1)
                        perm.extend(list(perm_sep_plane))
                        perm.extend(list(perm_side2))
                        self._safe_permutations.append(perm)
                        if add_opposite:
                            perm = list(perm_side2)
                            perm.extend(list(perm_sep_plane))
                            perm.extend(list(perm_side1))
                            self._safe_permutations.append(perm)
        return self._safe_permutations

    @property
    def as_dict(self):
        """
        Return the JSON-serializable dict representation of this SeparationPlane algorithm.

        Returns:
            dict: JSON-serializable representation of this SeparationPlane algorithm.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'plane_points': self.plane_points, 'mirror_plane': self.mirror_plane, 'ordered_plane': self.ordered_plane, 'point_groups': self.point_groups, 'ordered_point_groups': self.ordered_point_groups, 'explicit_permutations': [eperm.tolist() for eperm in self.explicit_permutations] if self.explicit_permutations is not None else None, 'explicit_optimized_permutations': [eoperm.tolist() for eoperm in self.explicit_optimized_permutations] if self.explicit_optimized_permutations is not None else None, 'multiplicity': self.multiplicity, 'other_plane_points': self.other_plane_points, 'minimum_number_of_points': self.minimum_number_of_points}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the SeparationPlane algorithm from its JSON-serializable dict representation.

        Args:
            dct: a JSON-serializable dict representation of an SeparationPlane algorithm.

        Returns:
            SeparationPlane: algorithm object
        """
        eop = [np.array(eo_perm) for eo_perm in dct.get('explicit_optimized_permutations', [])] or None
        return cls(plane_points=dct['plane_points'], mirror_plane=dct['mirror_plane'], ordered_plane=dct['ordered_plane'], point_groups=dct['point_groups'], ordered_point_groups=dct['ordered_point_groups'], explicit_permutations=[np.array(eperm) for eperm in dct['explicit_permutations']], explicit_optimized_permutations=eop, multiplicity=dct.get('multiplicity'), other_plane_points=dct.get('other_plane_points'), minimum_number_of_points=dct['minimum_number_of_points'])

    def __str__(self):
        out = 'Separation plane algorithm with the following reference separation :\n'
        out += f'[{'-'.join(map(str, [self.point_groups[0]]))}] | '
        out += f'[{'-'.join(map(str, [self.plane_points]))}] | '
        out += f'[{'-'.join(map(str, [self.point_groups[1]]))}]'
        return out