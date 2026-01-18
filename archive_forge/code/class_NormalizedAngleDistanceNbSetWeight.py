from __future__ import annotations
import abc
import os
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from monty.json import MSONable
from scipy.stats import gmean
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import EquivalentSiteSearchError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import get_lower_and_upper_f
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.func_utils import (
from pymatgen.core.operations import SymmOp
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class NormalizedAngleDistanceNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the normalized angle/distance."""
    SHORT_NAME = 'NormAngleDistWeight'

    def __init__(self, average_type, aa, bb):
        """Initialize NormalizedAngleDistanceNbSetWeight.

        Args:
            average_type: Average function.
            aa: Exponent for the angle values.
            bb: Exponent for the distance values.
        """
        self.average_type = average_type
        if self.average_type == 'geometric':
            self.eval = self.gweight
        elif self.average_type == 'arithmetic':
            self.eval = self.aweight
        else:
            raise ValueError(f"Average type is {average_type!r} while it should be 'geometric' or 'arithmetic'")
        self.aa = aa
        self.bb = bb
        if self.aa == 0:
            if self.bb == 1:
                self.fda = self.invdist
            elif self.bb == 0:
                raise ValueError('Both exponents are 0.')
            else:
                self.fda = self.invndist
        elif self.bb == 0:
            if self.aa == 1:
                self.fda = self.ang
            else:
                self.fda = self.angn
        elif self.aa == 1:
            self.fda = self.anginvdist if self.bb == 1 else self.anginvndist
        elif self.bb == 1:
            self.fda = self.angninvdist
        else:
            self.fda = self.angninvndist

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('average_type', 'aa', 'bb')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        return all((getattr(self, attr) == getattr(other, attr) for attr in needed_attrs))

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'average_type': self.average_type, 'aa': self.aa, 'bb': self.bb}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of NormalizedAngleDistanceNbSetWeight.

        Returns:
            NormalizedAngleDistanceNbSetWeight.
        """
        return cls(average_type=dct['average_type'], aa=dct['aa'], bb=dct['bb'])

    @staticmethod
    def invdist(nb_set):
        """Inverse distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of inverse distances.
        """
        return [1 / dist for dist in nb_set.normalized_distances]

    def invndist(self, nb_set):
        """Inverse power distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of inverse power distances.
        """
        return [1 / dist ** self.bb for dist in nb_set.normalized_distances]

    @staticmethod
    def ang(nb_set):
        """Angle weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of angle weights.
        """
        return nb_set.normalized_angles

    def angn(self, nb_set):
        """Power angle weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of power angle weights.
        """
        return [ang ** self.aa for ang in nb_set.normalized_angles]

    @staticmethod
    def anginvdist(nb_set):
        """Angle/distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of angle/distance weights.
        """
        nangles = nb_set.normalized_angles
        return [nangles[ii] / dist for ii, dist in enumerate(nb_set.normalized_distances)]

    def anginvndist(self, nb_set):
        """Angle/power distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of angle/power distance weights.
        """
        nangles = nb_set.normalized_angles
        return [nangles[ii] / dist ** self.bb for ii, dist in enumerate(nb_set.normalized_distances)]

    def angninvdist(self, nb_set):
        """Power angle/distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of power angle/distance weights.
        """
        nangles = nb_set.normalized_angles
        return [nangles[ii] ** self.aa / dist for ii, dist in enumerate(nb_set.normalized_distances)]

    def angninvndist(self, nb_set):
        """Power angle/power distance weight.

        Args:
            nb_set: Neighbors set.

        Returns:
            List of power angle/power distance weights.
        """
        nangles = nb_set.normalized_angles
        return [nangles[ii] ** self.aa / dist ** self.bb for ii, dist in enumerate(nb_set.normalized_distances)]

    def weight(self, nb_set, structure_environments, cn_map=None, additional_info=None):
        """Get the weight of a given neighbors set.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments used to estimate weight.
            cn_map: Mapping index for this neighbors set.
            additional_info: Additional information.

        Returns:
            Weight of the neighbors set.
        """
        fda_list = self.fda(nb_set=nb_set)
        return self.eval(fda_list=fda_list)

    @staticmethod
    def gweight(fda_list):
        """Geometric mean of the weights.

        Args:
            fda_list: List of estimator weights for each neighbor.

        Returns:
            Geometric mean of the weights.
        """
        return gmean(fda_list)

    @staticmethod
    def aweight(fda_list):
        """Standard mean of the weights.

        Args:
            fda_list: List of estimator weights for each neighbor.

        Returns:
            Standard mean of the weights.
        """
        return np.mean(fda_list)