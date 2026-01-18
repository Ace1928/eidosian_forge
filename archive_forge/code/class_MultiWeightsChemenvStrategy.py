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
class MultiWeightsChemenvStrategy(WeightedNbSetChemenvStrategy):
    """MultiWeightsChemenvStrategy."""
    STRATEGY_DESCRIPTION = 'Multi Weights ChemenvStrategy'
    DEFAULT_CE_ESTIMATOR = dict(function='power2_inverse_power2_decreasing', options={'max_csm': 8.0})

    def __init__(self, structure_environments=None, additional_condition=AbstractChemenvStrategy.AC.ONLY_ACB, symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE, dist_ang_area_weight=None, self_csm_weight=None, delta_csm_weight=None, cn_bias_weight=None, angle_weight=None, normalized_angle_distance_weight=None, ce_estimator=DEFAULT_CE_ESTIMATOR):
        """
        Constructor for the MultiWeightsChemenvStrategy.

        Args:
            structure_environments: StructureEnvironments object containing all the information on the
                coordination of the sites in a structure.
        """
        self._additional_condition = additional_condition
        self.dist_ang_area_weight = dist_ang_area_weight
        self.angle_weight = angle_weight
        self.normalized_angle_distance_weight = normalized_angle_distance_weight
        self.self_csm_weight = self_csm_weight
        self.delta_csm_weight = delta_csm_weight
        self.cn_bias_weight = cn_bias_weight
        self.ordered_weights = []
        nb_sets_weights = []
        if dist_ang_area_weight is not None:
            self.ordered_weights.append({'weight': dist_ang_area_weight, 'name': 'DistAngArea'})
            nb_sets_weights.append(dist_ang_area_weight)
        if self_csm_weight is not None:
            self.ordered_weights.append({'weight': self_csm_weight, 'name': 'SelfCSM'})
            nb_sets_weights.append(self_csm_weight)
        if delta_csm_weight is not None:
            self.ordered_weights.append({'weight': delta_csm_weight, 'name': 'DeltaCSM'})
            nb_sets_weights.append(delta_csm_weight)
        if cn_bias_weight is not None:
            self.ordered_weights.append({'weight': cn_bias_weight, 'name': 'CNBias'})
            nb_sets_weights.append(cn_bias_weight)
        if angle_weight is not None:
            self.ordered_weights.append({'weight': angle_weight, 'name': 'Angle'})
            nb_sets_weights.append(angle_weight)
        if normalized_angle_distance_weight is not None:
            self.ordered_weights.append({'weight': normalized_angle_distance_weight, 'name': 'NormalizedAngDist'})
            nb_sets_weights.append(normalized_angle_distance_weight)
        self.ce_estimator = ce_estimator
        self.ce_estimator_ratio_function = CSMInfiniteRatioFunction.from_dict(self.ce_estimator)
        self.ce_estimator_fractions = self.ce_estimator_ratio_function.fractions
        WeightedNbSetChemenvStrategy.__init__(self, structure_environments, additional_condition=additional_condition, symmetry_measure_type=symmetry_measure_type, nb_set_weights=nb_sets_weights, ce_estimator=ce_estimator)

    @classmethod
    def stats_article_weights_parameters(cls):
        """Initialize strategy used in the statistics article."""
        self_csm_weight = SelfCSMNbSetWeight(weight_estimator={'function': 'power2_decreasing_exp', 'options': {'max_csm': 8.0, 'alpha': 1}})
        surface_definition = {'type': 'standard_elliptic', 'distance_bounds': {'lower': 1.15, 'upper': 2.0}, 'angle_bounds': {'lower': 0.05, 'upper': 0.75}}
        da_area_weight = DistanceAngleAreaNbSetWeight(weight_type='has_intersection', surface_definition=surface_definition, nb_sets_from_hints='fallback_to_source', other_nb_sets='0_weight', additional_condition=DistanceAngleAreaNbSetWeight.AC.ONLY_ACB)
        symmetry_measure_type = 'csm_wcs_ctwcc'
        delta_weight = DeltaCSMNbSetWeight.delta_cn_specifics()
        bias_weight = angle_weight = nad_weight = None
        return cls(dist_ang_area_weight=da_area_weight, self_csm_weight=self_csm_weight, delta_csm_weight=delta_weight, cn_bias_weight=bias_weight, angle_weight=angle_weight, normalized_angle_distance_weight=nad_weight, symmetry_measure_type=symmetry_measure_type)

    @property
    def uniquely_determines_coordination_environments(self):
        """Whether this strategy uniquely determines coordination environments."""
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._additional_condition == other._additional_condition and self.symmetry_measure_type == other.symmetry_measure_type and (self.dist_ang_area_weight == other.dist_ang_area_weight) and (self.self_csm_weight == other.self_csm_weight) and (self.delta_csm_weight == other.delta_csm_weight) and (self.cn_bias_weight == other.cn_bias_weight) and (self.angle_weight == other.angle_weight) and (self.normalized_angle_distance_weight == other.normalized_angle_distance_weight) and (self.ce_estimator == other.ce_estimator)

    def as_dict(self):
        """
        Returns:
            Bson-serializable dict representation of the MultiWeightsChemenvStrategy object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'additional_condition': self._additional_condition, 'symmetry_measure_type': self.symmetry_measure_type, 'dist_ang_area_weight': self.dist_ang_area_weight.as_dict() if self.dist_ang_area_weight is not None else None, 'self_csm_weight': self.self_csm_weight.as_dict() if self.self_csm_weight is not None else None, 'delta_csm_weight': self.delta_csm_weight.as_dict() if self.delta_csm_weight is not None else None, 'cn_bias_weight': self.cn_bias_weight.as_dict() if self.cn_bias_weight is not None else None, 'angle_weight': self.angle_weight.as_dict() if self.angle_weight is not None else None, 'normalized_angle_distance_weight': self.normalized_angle_distance_weight.as_dict() if self.normalized_angle_distance_weight is not None else None, 'ce_estimator': self.ce_estimator}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the MultiWeightsChemenvStrategy object from a dict representation of the
        MultipleAbundanceChemenvStrategy object created using the as_dict method.

        Args:
            dct: dict representation of the MultiWeightsChemenvStrategy object

        Returns:
            MultiWeightsChemenvStrategy object.
        """
        if dct['normalized_angle_distance_weight'] is not None:
            nad_w = NormalizedAngleDistanceNbSetWeight.from_dict(dct['normalized_angle_distance_weight'])
        else:
            nad_w = None
        return cls(additional_condition=dct['additional_condition'], symmetry_measure_type=dct['symmetry_measure_type'], dist_ang_area_weight=DistanceAngleAreaNbSetWeight.from_dict(dct['dist_ang_area_weight']) if dct['dist_ang_area_weight'] is not None else None, self_csm_weight=SelfCSMNbSetWeight.from_dict(dct['self_csm_weight']) if dct['self_csm_weight'] is not None else None, delta_csm_weight=DeltaCSMNbSetWeight.from_dict(dct['delta_csm_weight']) if dct['delta_csm_weight'] is not None else None, cn_bias_weight=CNBiasNbSetWeight.from_dict(dct['cn_bias_weight']) if dct['cn_bias_weight'] is not None else None, angle_weight=AngleNbSetWeight.from_dict(dct['angle_weight']) if dct['angle_weight'] is not None else None, normalized_angle_distance_weight=nad_w, ce_estimator=dct['ce_estimator'])