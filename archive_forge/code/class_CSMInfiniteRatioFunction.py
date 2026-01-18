from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar
import numpy as np
from pymatgen.analysis.chemenv.utils.math_utils import (
class CSMInfiniteRatioFunction(AbstractRatioFunction):
    """Concrete implementation of a series of ratio functions applied to the continuous symmetry measure (CSM).

    Uses "infinite" ratio functions.

    See the following reference for details:
    ChemEnv: a fast and robust coordination environment identification tool,
    D. Waroquiers et al., Acta Cryst. B 76, 683 (2020).
    """
    ALLOWED_FUNCTIONS = dict(power2_inverse_decreasing=['max_csm'], power2_inverse_power2_decreasing=['max_csm'])

    def power2_inverse_decreasing(self, vals):
        """Get the evaluation of the ratio function f(x)=(x-1)^2 / x.

        The CSM values (i.e. "x"), are scaled to the "max_csm" parameter. The "a" constant
        correspond to the "alpha" parameter.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
        return power2_inverse_decreasing(vals, edges=[0.0, self.__dict__['max_csm']])

    def power2_inverse_power2_decreasing(self, vals):
        """Get the evaluation of the ratio function f(x)=(x-1)^2 / x^2.

        The CSM values (i.e. "x"), are scaled to the "max_csm" parameter. The "a" constant
        correspond to the "alpha" parameter.

        Args:
            vals: CSM values for which the ratio function has to be evaluated.

        Returns:
            Result of the ratio function applied to the CSM values.
        """
        return power2_inverse_power2_decreasing(vals, edges=[0.0, self.__dict__['max_csm']])

    def fractions(self, data):
        """Get the fractions from the CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate fractions.

        Returns:
            Corresponding fractions for each CSM.
        """
        if len(data) == 0:
            return None
        close_to_zero = np.isclose(data, 0.0, atol=1e-10).tolist()
        n_zeros = close_to_zero.count(True)
        if n_zeros == 1:
            fractions = [0.0] * len(data)
            fractions[close_to_zero.index(True)] = 1.0
            return fractions
        if n_zeros > 1:
            raise RuntimeError('Should not have more than one continuous symmetry measure with value equal to 0.0')
        fractions = self.eval(np.array(data))
        total = np.sum(fractions)
        if total > 0.0:
            return fractions / total
        return None

    def mean_estimator(self, data):
        """Get the weighted CSM using this CSM ratio function applied to the data.

        Args:
            data: List of CSM values to estimate the weighted CSM.

        Returns:
            Weighted CSM from this ratio function.
        """
        if len(data) == 0:
            return None
        if len(data) == 1:
            return data[0]
        fractions = self.fractions(data)
        if fractions is None:
            return None
        return np.sum(np.array(fractions) * np.array(data))
    ratios = fractions