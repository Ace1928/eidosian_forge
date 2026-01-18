import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_b(self, D):
    """coefficients for random effects/coefficients
        Display (3.4) from Laird, Lange, Stram (see help(Unit))

        D Z' W r
        """
    self.b = np.dot(D, np.dot(np.dot(self.Z.T, self.W), self.r))