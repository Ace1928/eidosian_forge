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
class CSMFloat(float, StrategyOption):
    """Real number representing a Continuous Symmetry Measure."""
    allowed_values = 'Real number between 0 and 100'

    def __new__(cls, cutoff) -> Self:
        """Special float that should be between 0 and 100.

        Args:
            cutoff: CSM.
        """
        flt = float.__new__(cls, cutoff)
        if not 0 <= flt <= 100:
            raise ValueError(f'Continuous symmetry measure limits should be between 0 and 100, got {flt}')
        return flt

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'value': self}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize CSM from dict.

        Args:
           dct (dict): Dict representation of the CSM.
        """
        return cls(dct['value'])