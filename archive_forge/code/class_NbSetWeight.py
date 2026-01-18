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
class NbSetWeight(MSONable, abc.ABC):
    """Abstract object for neighbors sets weights estimations."""

    @abc.abstractmethod
    def as_dict(self):
        """A JSON-serializable dict representation of this neighbors set weight."""

    @abc.abstractmethod
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