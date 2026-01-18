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
class DistanceAngleAreaNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the area in the distance-angle space."""
    SHORT_NAME = 'DistAngleAreaWeight'
    AC = AdditionalConditions()
    DEFAULT_SURFACE_DEFINITION = dict(type='standard_elliptic', distance_bounds={'lower': 1.2, 'upper': 1.8}, angle_bounds={'lower': 0.1, 'upper': 0.8})

    def __init__(self, weight_type='has_intersection', surface_definition=DEFAULT_SURFACE_DEFINITION, nb_sets_from_hints='fallback_to_source', other_nb_sets='0_weight', additional_condition=AC.ONLY_ACB, smoothstep_distance=None, smoothstep_angle=None):
        """Initialize CNBiasNbSetWeight.

        Args:
            weight_type: Type of weight.
            surface_definition: Definition of the surface.
            nb_sets_from_hints: How to deal with neighbors sets obtained from "hints".
            other_nb_sets: What to do with other neighbors sets.
            additional_condition: Additional condition to be used.
            smoothstep_distance: Smoothstep distance.
            smoothstep_angle: Smoothstep angle.
        """
        self.weight_type = weight_type
        if weight_type == 'has_intersection':
            self.area_weight = self.w_area_has_intersection
        elif weight_type == 'has_intersection_smoothstep':
            raise NotImplementedError
        else:
            raise ValueError(f'Weight type is {weight_type!r} while it should be "has_intersection"')
        self.surface_definition = surface_definition
        self.nb_sets_from_hints = nb_sets_from_hints
        self.other_nb_sets = other_nb_sets
        self.additional_condition = additional_condition
        self.smoothstep_distance = smoothstep_distance
        self.smoothstep_angle = smoothstep_angle
        if self.nb_sets_from_hints == 'fallback_to_source':
            if self.other_nb_sets == '0_weight':
                self.w_area_intersection_specific = self.w_area_intersection_nbsfh_fbs_onb0
            else:
                raise ValueError('Other nb_sets should be "0_weight"')
        else:
            raise ValueError('Nb_sets from hints should fallback to source')
        lower_and_upper_functions = get_lower_and_upper_f(surface_calculation_options=surface_definition)
        self.dmin = surface_definition['distance_bounds']['lower']
        self.dmax = surface_definition['distance_bounds']['upper']
        self.amin = surface_definition['angle_bounds']['lower']
        self.amax = surface_definition['angle_bounds']['upper']
        self.f_lower = lower_and_upper_functions['lower']
        self.f_upper = lower_and_upper_functions['upper']

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
        return self.area_weight(nb_set=nb_set, structure_environments=structure_environments, cn_map=cn_map, additional_info=additional_info)

    def w_area_has_intersection(self, nb_set, structure_environments, cn_map, additional_info):
        """Get intersection of the neighbors set area with the surface.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments.
            cn_map: Mapping index of the neighbors set.
            additional_info: Additional information.

        Returns:
            Area intersection between neighbors set and surface.
        """
        return self.w_area_intersection_specific(nb_set=nb_set, structure_environments=structure_environments, cn_map=cn_map, additional_info=additional_info)

    def w_area_intersection_nbsfh_fbs_onb0(self, nb_set, structure_environments, cn_map, additional_info):
        """Get intersection of the neighbors set area with the surface.

        Args:
            nb_set: Neighbors set.
            structure_environments: Structure environments.
            cn_map: Mapping index of the neighbors set.
            additional_info: Additional information.

        Returns:
            Area intersection between neighbors set and surface.
        """
        dist_ang_sources = [src for src in nb_set.sources if src['origin'] == 'dist_ang_ac_voronoi' and src['ac'] == self.additional_condition]
        if len(dist_ang_sources) > 0:
            for src in dist_ang_sources:
                d1 = src['dp_dict']['min']
                d2 = src['dp_dict']['next']
                a1 = src['ap_dict']['next']
                a2 = src['ap_dict']['max']
                if self.rectangle_crosses_area(d1=d1, d2=d2, a1=a1, a2=a2):
                    return 1
            return 0
        from_hints_sources = [src for src in nb_set.sources if src['origin'] == 'nb_set_hints']
        if len(from_hints_sources) == 0:
            return 0
        if len(from_hints_sources) != 1:
            raise ValueError('Found multiple hints sources for nb_set')
        cn_map_src = from_hints_sources[0]['cn_map_source']
        nb_set_src = structure_environments.neighbors_sets[nb_set.isite][cn_map_src[0]][cn_map_src[1]]
        dist_ang_sources = [src for src in nb_set_src.sources if src['origin'] == 'dist_ang_ac_voronoi' and src['ac'] == self.additional_condition]
        if len(dist_ang_sources) == 0:
            return 0
        for src in dist_ang_sources:
            d1 = src['dp_dict']['min']
            d2 = src['dp_dict']['next']
            a1 = src['ap_dict']['next']
            a2 = src['ap_dict']['max']
            if self.rectangle_crosses_area(d1=d1, d2=d2, a1=a1, a2=a2):
                return 1
        return 0

    def rectangle_crosses_area(self, d1, d2, a1, a2):
        """Whether a given rectangle crosses the area defined by the upper and lower curves.

        Args:
            d1: lower d.
            d2: upper d.
            a1: lower a.
            a2: upper a.
        """
        if d1 <= self.dmin and d2 <= self.dmin:
            return False
        if d1 >= self.dmax and d2 >= self.dmax:
            return False
        if d1 <= self.dmin and d2 <= self.dmax:
            ld2 = self.f_lower(d2)
            if a2 <= ld2 or a1 >= self.amax:
                return False
            return True
        if d1 <= self.dmin and d2 >= self.dmax:
            if a2 <= self.amin or a1 >= self.amax:
                return False
            return True
        if self.dmin <= d1 <= self.dmax and self.dmin <= d2 <= self.dmax:
            ld1 = self.f_lower(d1)
            ld2 = self.f_lower(d2)
            if a2 <= ld1 and a2 <= ld2:
                return False
            ud1 = self.f_upper(d1)
            ud2 = self.f_upper(d2)
            if a1 >= ud1 and a1 >= ud2:
                return False
            return True
        if self.dmin <= d1 <= self.dmax and d2 >= self.dmax:
            ud1 = self.f_upper(d1)
            if a1 >= ud1 or a2 <= self.amin:
                return False
            return True
        raise ValueError('Should not reach this point!')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.weight_type == other.weight_type and self.surface_definition == other.surface_definition and (self.nb_sets_from_hints == other.nb_sets_from_hints) and (self.other_nb_sets == other.other_nb_sets) and (self.additional_condition == other.additional_condition)

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'weight_type': self.weight_type, 'surface_definition': self.surface_definition, 'nb_sets_from_hints': self.nb_sets_from_hints, 'other_nb_sets': self.other_nb_sets, 'additional_condition': self.additional_condition}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of DistanceAngleAreaNbSetWeight.

        Returns:
            DistanceAngleAreaNbSetWeight.
        """
        return cls(weight_type=dct['weight_type'], surface_definition=dct['surface_definition'], nb_sets_from_hints=dct['nb_sets_from_hints'], other_nb_sets=dct['other_nb_sets'], additional_condition=dct['additional_condition'])