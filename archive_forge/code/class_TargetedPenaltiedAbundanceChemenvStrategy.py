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
class TargetedPenaltiedAbundanceChemenvStrategy(SimpleAbundanceChemenvStrategy):
    """
    Simple ChemenvStrategy using the neighbors that are the most "abundant" in the grid of angle and distance
    parameters for the definition of neighbors in the Voronoi approach, with a bias for a given list of target
    environments. This can be useful in the case of, e.g. connectivity search of some given environment.
    The coordination environment is then given as the one with the lowest continuous symmetry measure.
    """
    DEFAULT_TARGET_ENVIRONMENTS = ('O:6',)

    def __init__(self, structure_environments=None, truncate_dist_ang=True, additional_condition=AbstractChemenvStrategy.AC.ONLY_ACB, max_nabundant=5, target_environments=DEFAULT_TARGET_ENVIRONMENTS, target_penalty_type='max_csm', max_csm=5.0, symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE):
        """Initialize strategy.

        Not yet implemented.

        Args:
            structure_environments:
            truncate_dist_ang:
            additional_condition:
            max_nabundant:
            target_environments:
            target_penalty_type:
            max_csm:
            symmetry_measure_type:
        """
        raise NotImplementedError('TargetedPenaltiedAbundanceChemenvStrategy not yet implemented')
        super().__init__(self, structure_environments, additional_condition=additional_condition, symmetry_measure_type=symmetry_measure_type)
        self.max_nabundant = max_nabundant
        self.target_environments = target_environments
        self.target_penalty_type = target_penalty_type
        self.max_csm = max_csm

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
        chemical_environments = self.structure_environments.ce_list[self.structure_environments.sites_map[isite]][cn_map[0]][cn_map[1]]
        if return_map:
            if chemical_environments.coord_geoms is None or len(chemical_environments) == 0:
                return (cn_map[0], cn_map)
            return (chemical_environments.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type), cn_map)
        if chemical_environments.coord_geoms is None:
            return cn_map[0]
        return chemical_environments.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type)

    def _get_map(self, isite):
        maps_and_surfaces = SimpleAbundanceChemenvStrategy._get_maps_surfaces(self, isite)
        if maps_and_surfaces is None:
            return SimpleAbundanceChemenvStrategy._get_map(self, isite)
        current_map = None
        current_target_env_csm = 100
        surfaces = [map_and_surface['surface'] for map_and_surface in maps_and_surfaces]
        order = np.argsort(surfaces)[::-1]
        target_cgs = [AllCoordinationGeometries().get_geometry_from_mp_symbol(mp_symbol) for mp_symbol in self.target_environments]
        target_cns = [cg.coordination_number for cg in target_cgs]
        for ii in range(min([len(maps_and_surfaces), self.max_nabundant])):
            my_map_and_surface = maps_and_surfaces[order[ii]]
            my_map = my_map_and_surface['map']
            cn = my_map[0]
            if cn not in target_cns or cn > 12 or cn == 0:
                continue
            all_conditions = [params[2] for params in my_map_and_surface['parameters_indices']]
            if self._additional_condition not in all_conditions:
                continue
            cg, cgdict = self.structure_environments.ce_list[self.structure_environments.sites_map[isite]][my_map[0]][my_map[1]].minimum_geometry(symmetry_measure_type=self._symmetry_measure_type)
            if cg in self.target_environments and cgdict['symmetry_measure'] <= self.max_csm and (cgdict['symmetry_measure'] < current_target_env_csm):
                current_map = my_map
                current_target_env_csm = cgdict['symmetry_measure']
        if current_map is not None:
            return current_map
        return SimpleAbundanceChemenvStrategy._get_map(self, isite)

    @property
    def uniquely_determines_coordination_environments(self):
        """Whether this strategy uniquely determines coordination environments."""
        return True

    def as_dict(self):
        """
        Bson-serializable dict representation of the TargetedPenaltiedAbundanceChemenvStrategy object.

        Returns:
            Bson-serializable dict representation of the TargetedPenaltiedAbundanceChemenvStrategy object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'additional_condition': self._additional_condition, 'max_nabundant': self.max_nabundant, 'target_environments': self.target_environments, 'target_penalty_type': self.target_penalty_type, 'max_csm': self.max_csm}

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.additional_condition == other.additional_condition and self.max_nabundant == other.max_nabundant and (self.target_environments == other.target_environments) and (self.target_penalty_type == other.target_penalty_type) and (self.max_csm == other.max_csm)

    @classmethod
    def from_dict(cls, dct) -> Self:
        """
        Reconstructs the TargetedPenaltiedAbundanceChemenvStrategy object from a dict representation of the
        TargetedPenaltiedAbundanceChemenvStrategy object created using the as_dict method.

        Args:
            dct: dict representation of the TargetedPenaltiedAbundanceChemenvStrategy object

        Returns:
            TargetedPenaltiedAbundanceChemenvStrategy object.
        """
        return cls(additional_condition=dct['additional_condition'], max_nabundant=dct['max_nabundant'], target_environments=dct['target_environments'], target_penalty_type=dct['target_penalty_type'], max_csm=dct['max_csm'])