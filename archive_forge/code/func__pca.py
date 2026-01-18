import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _pca(self):
    """
        Main PCA routine
        """
    self._compute_eig()
    self._compute_pca_from_eig()
    self.projection = self.project()