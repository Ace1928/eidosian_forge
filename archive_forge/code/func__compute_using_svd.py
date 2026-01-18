import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_using_svd(self):
    """SVD method to compute eigenvalues and eigenvecs"""
    x = self.transformed_data
    u, s, v = np.linalg.svd(x, full_matrices=self._svd_full_matrices)
    self.eigenvals = s ** 2.0
    self.eigenvecs = v.T