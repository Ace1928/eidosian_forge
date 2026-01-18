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
class SimplestChemenvStrategy(AbstractChemenvStrategy):
    """
    Simplest ChemenvStrategy using fixed angle and distance parameters for the definition of neighbors in the
    Voronoi approach. The coordination environment is then given as the one with the lowest continuous symmetry measure.
    """
    DEFAULT_DISTANCE_CUTOFF = 1.4
    DEFAULT_ANGLE_CUTOFF = 0.3
    DEFAULT_CONTINUOUS_SYMMETRY_MEASURE_CUTOFF = 10
    DEFAULT_ADDITIONAL_CONDITION = AbstractChemenvStrategy.AC.ONLY_ACB
    STRATEGY_OPTIONS: ClassVar[dict[str, dict]] = dict(distance_cutoff=dict(type=DistanceCutoffFloat, internal='_distance_cutoff', default=DEFAULT_DISTANCE_CUTOFF), angle_cutoff=dict(type=AngleCutoffFloat, internal='_angle_cutoff', default=DEFAULT_ANGLE_CUTOFF), additional_condition=dict(type=AdditionalConditionInt, internal='_additional_condition', default=DEFAULT_ADDITIONAL_CONDITION), continuous_symmetry_measure_cutoff=dict(type=CSMFloat, internal='_continuous_symmetry_measure_cutoff', default=DEFAULT_CONTINUOUS_SYMMETRY_MEASURE_CUTOFF))
    STRATEGY_DESCRIPTION = 'Simplest ChemenvStrategy using fixed angle and distance parameters \nfor the definition of neighbors in the Voronoi approach. \nThe coordination environment is then given as the one with the \nlowest continuous symmetry measure.'

    def __init__(self, structure_environments=None, distance_cutoff=DEFAULT_DISTANCE_CUTOFF, angle_cutoff=DEFAULT_ANGLE_CUTOFF, additional_condition=DEFAULT_ADDITIONAL_CONDITION, continuous_symmetry_measure_cutoff=DEFAULT_CONTINUOUS_SYMMETRY_MEASURE_CUTOFF, symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE):
        """
        Constructor for this SimplestChemenvStrategy.

        Args:
            distance_cutoff: Distance cutoff used
            angle_cutoff: Angle cutoff used.
        """
        AbstractChemenvStrategy.__init__(self, structure_environments, symmetry_measure_type=symmetry_measure_type)
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
        self.additional_condition = additional_condition
        self.continuous_symmetry_measure_cutoff = continuous_symmetry_measure_cutoff

    @property
    def uniquely_determines_coordination_environments(self):
        """Whether this strategy uniquely determines coordination environments."""
        return True

    @property
    def distance_cutoff(self):
        """Distance cutoff used."""
        return self._distance_cutoff

    @distance_cutoff.setter
    def distance_cutoff(self, distance_cutoff):
        """Set the distance cutoff for this strategy.

        Args:
            distance_cutoff: Distance cutoff.
        """
        self._distance_cutoff = DistanceCutoffFloat(distance_cutoff)

    @property
    def angle_cutoff(self):
        """Angle cutoff used."""
        return self._angle_cutoff

    @angle_cutoff.setter
    def angle_cutoff(self, angle_cutoff):
        """Set the angle cutoff for this strategy.

        Args:
            angle_cutoff: Angle cutoff.
        """
        self._angle_cutoff = AngleCutoffFloat(angle_cutoff)

    @property
    def additional_condition(self):
        """Additional condition for this strategy."""
        return self._additional_condition

    @additional_condition.setter
    def additional_condition(self, additional_condition):
        """Set the additional condition for this strategy.

        Args:
            additional_condition: Additional condition.
        """
        self._additional_condition = AdditionalConditionInt(additional_condition)

    @property
    def continuous_symmetry_measure_cutoff(self):
        """CSM cutoff used."""
        return self._continuous_symmetry_measure_cutoff

    @continuous_symmetry_measure_cutoff.setter
    def continuous_symmetry_measure_cutoff(self, continuous_symmetry_measure_cutoff):
        """Set the CSM cutoff for this strategy.

        Args:
            continuous_symmetry_measure_cutoff: CSM cutoff
        """
        self._continuous_symmetry_measure_cutoff = CSMFloat(continuous_symmetry_measure_cutoff)

    def get_site_neighbors(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None):
        """Get the neighbors of a given site.

        Args:
            site: Site for which neighbors are needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.

        Returns:
            List of coordinated neighbors of site.
        """
        if isite is None:
            isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
        _ce, cn_map = self.get_site_coordination_environment(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_map=True)
        nb_set = self.structure_environments.neighbors_sets[isite][cn_map[0]][cn_map[1]]
        eq_site_ps = nb_set.neighb_sites
        coordinated_neighbors = []
        for ps in eq_site_ps:
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
        neighbors_normalized_distances = self.structure_environments.voronoi.neighbors_normalized_distances[isite]
        neighbors_normalized_angles = self.structure_environments.voronoi.neighbors_normalized_angles[isite]
        i_dist = None
        for iwd, wd in enumerate(neighbors_normalized_distances):
            if self.distance_cutoff >= wd['min']:
                i_dist = iwd
            else:
                break
        i_ang = None
        for iwa, wa in enumerate(neighbors_normalized_angles):
            if self.angle_cutoff <= wa['max']:
                i_ang = iwa
            else:
                break
        if i_dist is None or i_ang is None:
            raise ValueError('Distance or angle parameter not found ...')
        my_cn = my_inb_set = None
        found = False
        for cn, nb_sets in self.structure_environments.neighbors_sets[isite].items():
            for inb_set, nb_set in enumerate(nb_sets):
                sources = [src for src in nb_set.sources if src['origin'] == 'dist_ang_ac_voronoi' and src['ac'] == self.additional_condition]
                for src in sources:
                    if src['idp'] == i_dist and src['iap'] == i_ang:
                        my_cn = cn
                        my_inb_set = inb_set
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None
        cn_map = (my_cn, my_inb_set)
        ce = self.structure_environments.ce_list[self.structure_environments.sites_map[isite]][cn_map[0]][cn_map[1]]
        if ce is None:
            return None
        coord_geoms = ce.coord_geoms
        if return_map:
            if coord_geoms is None:
                return (cn_map[0], cn_map)
            return (ce.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type), cn_map)
        if coord_geoms is None:
            return cn_map[0]
        return ce.minimum_geometry(symmetry_measure_type=self._symmetry_measure_type)

    def get_site_coordination_environments_fractions(self, site, isite=None, dequivsite=None, dthissite=None, mysym=None, ordered=True, min_fraction=0, return_maps=True, return_strategy_dict_info=False):
        """Get the coordination environments of a given site and additional information.

        Args:
            site: Site for which coordination environment is needed.
            isite: Index of the site.
            dequivsite: Translation of the equivalent site.
            dthissite: Translation of this site.
            mysym: Symmetry to be applied.
            ordered: Whether to order the list by fractions.
            min_fraction: Minimum fraction to include in the list
            return_maps: Whether to return cn_maps (identifies all the NeighborsSet used).
            return_strategy_dict_info: Whether to add the info about the strategy used.

        Returns:
            List of Dict with coordination environment, fraction and additional info.
        """
        if isite is None or dequivsite is None or dthissite is None or (mysym is None):
            isite, dequivsite, dthissite, mysym = self.equivalent_site_index_and_transform(site)
        site_nb_sets = self.structure_environments.neighbors_sets[isite]
        if site_nb_sets is None:
            return None
        ce_and_map = self.get_site_coordination_environment(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_map=True)
        if ce_and_map is None:
            return None
        ce, ce_map = ce_and_map
        if ce is None:
            ce_dict = {'ce_symbol': f'UNKNOWN:{ce_map[0]}', 'ce_dict': None, 'ce_fraction': 1}
        else:
            ce_dict = {'ce_symbol': ce[0], 'ce_dict': ce[1], 'ce_fraction': 1}
        if return_maps:
            ce_dict['ce_map'] = ce_map
        if return_strategy_dict_info:
            ce_dict['strategy_info'] = {}
        return [ce_dict]

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
        env = self.get_site_coordination_environment(site=site, isite=isite, dequivsite=dequivsite, dthissite=dthissite, mysym=mysym, return_map=return_maps)
        return [env]

    def add_strategy_visualization_to_subplot(self, subplot, visualization_options=None, plot_type=None):
        """Add a visual of the strategy on a distance-angle plot.

        Args:
            subplot: Axes object onto the visual should be added.
            visualization_options: Options for the visual.
            plot_type: Type of distance-angle plot.
        """
        subplot.plot(self._distance_cutoff, self._angle_cutoff, 'o', markeredgecolor=None, markerfacecolor='w', markersize=12)
        subplot.plot(self._distance_cutoff, self._angle_cutoff, 'x', linewidth=2, markersize=12)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._distance_cutoff == other._distance_cutoff and self._angle_cutoff == other._angle_cutoff and (self._additional_condition == other._additional_condition) and (self._continuous_symmetry_measure_cutoff == other._continuous_symmetry_measure_cutoff) and (self.symmetry_measure_type == other.symmetry_measure_type)

    def as_dict(self):
        """
        Bson-serializable dict representation of the SimplestChemenvStrategy object.

        Returns:
            Bson-serializable dict representation of the SimplestChemenvStrategy object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'distance_cutoff': float(self._distance_cutoff), 'angle_cutoff': float(self._angle_cutoff), 'additional_condition': int(self._additional_condition), 'continuous_symmetry_measure_cutoff': float(self._continuous_symmetry_measure_cutoff), 'symmetry_measure_type': self._symmetry_measure_type}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the SimplestChemenvStrategy object from a dict representation of the SimplestChemenvStrategy object
        created using the as_dict method.

        Args:
            dct: dict representation of the SimplestChemenvStrategy object

        Returns:
            StructureEnvironments object.
        """
        return cls(distance_cutoff=dct['distance_cutoff'], angle_cutoff=dct['angle_cutoff'], additional_condition=dct['additional_condition'], continuous_symmetry_measure_cutoff=dct['continuous_symmetry_measure_cutoff'], symmetry_measure_type=dct['symmetry_measure_type'])