import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_r(self, alpha):
    """residual after removing fixed effects

        Display (3.5) from Laird, Lange, Stram (see help(Unit))
        """
    self.r = self.Y - np.dot(self.X, alpha)