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
class SimpleAbundanceChemenvStrategy(AbstractChemenvStrategy):
    """
    Simple ChemenvStrategy using the neighbors that are the most "abundant" in the grid of angle and distance
    parameters for the definition of neighbors in the Voronoi approach.
    The coordination environment is then given as the one with the lowest continuous symmetry measure.
    """
    DEFAULT_MAX_DIST = 2.0
    DEFAULT_ADDITIONAL_CONDITION = AbstractChemenvStrategy.AC.ONLY_ACB
    STRATEGY_OPTIONS: ClassVar[dict[str, dict]] = dict(surface_calculation_type={}, additional_condition=dict(type=AdditionalConditionInt, internal='_additional_condition', default=DEFAULT_ADDITIONAL_CONDITION))
    STRATEGY_DESCRIPTION = 'Simple Abundance ChemenvStrategy using the most "abundant" neighbors map \nfor the definition of neighbors in the Voronoi approach. \nThe coordination environment is then given as the one with the \nlowest continuous symmetry measure.'

    def __init__(self, structure_environments=None, additional_condition=AbstractChemenvStrategy.AC.ONLY_ACB, symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE):
        """
        Constructor for the SimpleAbundanceChemenvStrategy.

        Args:
            structure_environments: StructureEnvironments object containing all the information on the
                coordination of the sites in a structure.
        """
        raise NotImplementedError('SimpleAbundanceChemenvStrategy not yet implemented')
        AbstractChemenvStrategy.__init__(self, structure_environments, symmetry_measure_type=symmetry_measure_type)
        self._additional_condition = additional_condition

    @property
    def uniquely_determines_coordination_environments(self):
        """Whether this strategy uniquely determines coordination environments."""
        return True

    def get_site_neighbors(self, site):
        """Get the neighbors of a given site with this strategy.

        Args:
            site: Periodic site.

        Returns:
            List of neighbors of site.
        """
        isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
        cn_map = self._get_map(isite)
        eqsite_ps = self.structure_environments.unique_coordinated_neighbors(isite, cn_map=cn_map)
        coordinated_neighbors = []
        for ps in eqsite_ps:
            coords = mysym.operate(ps.frac_coords + dequivsite) + dthissite
            ps_site = PeriodicSite(ps._species, coords, ps._lattice)
            coordinated_neighbors.append(ps_site)
        return coordinated_neighbors

    def get_site_coordination_environment(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, return_map=False):
        """Get the coordination environment of a given site.

        Args:
            site: Site for which coordination environment is needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.
            return_map: Whether to return cn_map (identifies the NeighborsSet used).

        Returns:
            Coordination environment of site.
        """
        if isite is None:
            isite, *_ = self.equivalent_site_index_and_transform(site)
        cn_map = self._get_map(isite)
        if cn_map is None:
            return None
        coord_geoms = self.structure_environments.ce_list[self.structure_environments.sites_map[isite]][cn_map[0]][cn_map[1]]
        if return_map:
            if coord_geoms is None:
                return (cn_map[0], cn_map)
            return (coord_geoms.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type), cn_map)
        if coord_geoms is None:
            return cn_map[0]
        return coord_geoms.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type)

    def get_site_coordination_environments(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, return_maps=False):
        """Get the coordination environments of a given site.

        Args:
            site: Site for which coordination environment is needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.
            return_maps: Whether to return cn_maps (identifies all the NeighborsSet used).

        Returns:
            List of coordination environment.
        """
        return [self.get_site_coordination_environment(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_map=return_maps)]

    def _get_map(self, isite):
        maps_and_surfaces = self._get_maps_surfaces(isite)
        if maps_and_surfaces is None:
            return None
        surface_max = 0
        imax = -1
        for ii, map_and_surface in enumerate(maps_and_surfaces):
            all_additional_conditions = [ac[2] for ac in map_and_surface['parameters_indices']]
            if self._additional_condition in all_additional_conditions and map_and_surface['surface'] > surface_max:
                surface_max = map_and_surface['surface']
                imax = ii
        return maps_and_surfaces[imax]['map']

    def _get_maps_surfaces(self, isite, surface_calculation_type=None):
        if surface_calculation_type is None:
            surface_calculation_type = {'distance_parameter': ('initial_normalized', None), 'angle_parameter': ('initial_normalized', None)}
        return self.structure_environments.voronoi.maps_and_surfaces(isite=isite, surface_calculation_type=surface_calculation_type, max_dist=self.DEFAULT_MAX_DIST)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._additional_condition == other.additional_condition

    def as_dict(self):
        """
        Bson-serializable dict representation of the SimpleAbundanceChemenvStrategy object.

        Returns:
            Bson-serializable dict representation of the SimpleAbundanceChemenvStrategy object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'additional_condition': self._additional_condition}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the SimpleAbundanceChemenvStrategy object from a dict representation of the
        SimpleAbundanceChemenvStrategy object created using the as_dict method.

        Args:
            dct: dict representation of the SimpleAbundanceChemenvStrategy object

        Returns:
            StructureEnvironments object.
        """
        return cls(additional_condition=dct['additional_condition'])