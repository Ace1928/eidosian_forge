import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_W(self):
    """inverse covariance of observations (nobs_i, nobs_i)  (JP check)
        Display (3.2) from Laird, Lange, Stram (see help(Unit))
        """
    self.W = L.inv(self.S)