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
class SelfCSMNbSetWeight(NbSetWeight):
    """Weight of neighbors set based on the Self CSM."""
    SHORT_NAME = 'SelfCSMWeight'
    DEFAULT_EFFECTIVE_CSM_ESTIMATOR = dict(function='power2_inverse_decreasing', options={'max_csm': 8.0})
    DEFAULT_WEIGHT_ESTIMATOR = dict(function='power2_decreasing_exp', options={'max_csm': 8.0, 'alpha': 1})
    DEFAULT_SYMMETRY_MEASURE_TYPE = 'csm_wcs_ctwcc'

    def __init__(self, effective_csm_estimator=DEFAULT_EFFECTIVE_CSM_ESTIMATOR, weight_estimator=DEFAULT_WEIGHT_ESTIMATOR, symmetry_measure_type=DEFAULT_SYMMETRY_MEASURE_TYPE):
        """Initialize SelfCSMNbSetWeight.

        Args:
            effective_csm_estimator: Ratio function used for the effective CSM (comparison between neighbors sets).
            weight_estimator: Weight estimator within a given neighbors set.
            symmetry_measure_type: Type of symmetry measure to be used.
        """
        self.effective_csm_estimator = effective_csm_estimator
        self.effective_csm_estimator_rf = CSMInfiniteRatioFunction.from_dict(effective_csm_estimator)
        self.weight_estimator = weight_estimator
        self.weight_estimator_rf = CSMFiniteRatioFunction.from_dict(weight_estimator)
        self.symmetry_measure_type = symmetry_measure_type
        self.max_effective_csm = self.effective_csm_estimator['options']['max_csm']

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
        effective_csm = get_effective_csm(nb_set=nb_set, cn_map=cn_map, structure_environments=structure_environments, additional_info=additional_info, symmetry_measure_type=self.symmetry_measure_type, max_effective_csm=self.max_effective_csm, effective_csm_estimator_ratio_function=self.effective_csm_estimator_rf)
        weight = self.weight_estimator_rf.evaluate(effective_csm)
        set_info(additional_info=additional_info, field='self_csms_weights', isite=nb_set.isite, cn_map=cn_map, value=weight)
        return weight

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.effective_csm_estimator == other.effective_csm_estimator and self.weight_estimator == other.weight_estimator and (self.symmetry_measure_type == other.symmetry_measure_type)

    def as_dict(self):
        """MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'effective_csm_estimator': self.effective_csm_estimator, 'weight_estimator': self.weight_estimator, 'symmetry_measure_type': self.symmetry_measure_type}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Initialize from dict.

        Args:
            dct (dict): Dict representation of SelfCSMNbSetWeight.

        Returns:
            SelfCSMNbSetWeight.
        """
        return cls(effective_csm_estimator=dct['effective_csm_estimator'], weight_estimator=dct['weight_estimator'], symmetry_measure_type=dct['symmetry_measure_type'])