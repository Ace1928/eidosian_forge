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
def _get_maps_surfaces(self, isite, surface_calculation_type=None):
    if surface_calculation_type is None:
        surface_calculation_type = {'distance_parameter': ('initial_normalized', None), 'angle_parameter': ('initial_normalized', None)}
    return self.structure_environments.voronoi.maps_and_surfaces(isite=isite, surface_calculation_type=surface_calculation_type, max_dist=self.DEFAULT_MAX_DIST)