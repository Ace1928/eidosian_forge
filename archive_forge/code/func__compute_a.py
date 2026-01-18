import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
def _compute_a(self):
    """fixed effects parameters

        Display (3.1) of
        Laird, Lange, Stram (see help(Mixed)).
        """
    for unit in self.units:
        unit.fit(self.a, self.D, self.sigma)
    S = sum([unit.compute_xtwx() for unit in self.units])
    Y = sum([unit.compute_xtwy() for unit in self.units])
    self.Sinv = L.pinv(S)
    self.a = np.dot(self.Sinv, Y)