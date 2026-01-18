from statsmodels.regression.linear_model import GLS
import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from scipy import sparse
class SysResults(LikelihoodModelResults):
    """
    Not implemented yet.
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        super().__init__(model, params, normalized_cov_params, scale)
        self._get_results()

    def _get_results(self):
        pass