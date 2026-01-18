from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def adapt_rho(self):
    """
        Adapt rho value based on current primal and dual residuals
        """
    rho_new = self.compute_rho_estimate()
    self.work.info.rho_estimate = rho_new
    adaptive_rho_tolerance = self.work.settings.adaptive_rho_tolerance
    if rho_new > adaptive_rho_tolerance * self.work.settings.rho or rho_new < 1.0 / adaptive_rho_tolerance * self.work.settings.rho:
        self.update_rho(rho_new)
        self.work.info.rho_updates += 1