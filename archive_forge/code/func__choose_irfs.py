import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def _choose_irfs(self, orth=False, svar=False):
    if orth:
        return self.orth_irfs
    elif svar:
        return self.svar_irfs
    else:
        return self.irfs