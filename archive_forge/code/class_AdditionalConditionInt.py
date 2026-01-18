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
class AdditionalConditionInt(int, StrategyOption):
    """Integer representing an additional condition in a strategy."""
    allowed_values = 'Integer amongst :\n'
    for integer, description in AdditionalConditions.CONDITION_DESCRIPTION.items():
        allowed_values += f' - {integer} for {description!r}\n'

    def __new__(cls, integer) -> Self:
        """Special int representing additional conditions."""
        if str(int(integer)) != str(integer):
            raise ValueError(f'Additional condition {integer} is not an integer')
        integer = int.__new__(cls, integer)
        if integer not in AdditionalConditions.ALL:
            raise ValueError(f'Additional condition {integer} is not allowed')
        return integer

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'value': self}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize additional condition from dict.

        Args:
           dct (dict): Dict representation of the additional condition.
        """
        return cls(dct['value'])