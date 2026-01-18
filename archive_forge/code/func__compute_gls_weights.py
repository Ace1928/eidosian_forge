import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_gls_weights(self):
    """
        Computes GLS weights based on percentage of data fit
        """
    projection = np.asarray(self.project(transform=False))
    errors = self.transformed_data - projection
    if self._ncomp == self._nvar:
        raise ValueError('gls can only be used when ncomp < nvar so that residuals have non-zero variance')
    var = (errors ** 2.0).mean(0)
    weights = 1.0 / var
    weights = weights / np.sqrt((weights ** 2.0).mean())
    nvar = self._nvar
    eff_series_perc = 1.0 / sum((weights / weights.sum()) ** 2.0) / nvar
    if eff_series_perc < 0.1:
        eff_series = int(np.round(eff_series_perc * nvar))
        import warnings
        warn = f'Many series are being down weighted by GLS. Of the {nvar} series, the GLS\nestimates are based on only {eff_series} (effective) series.'
        warnings.warn(warn, EstimationWarning)
    self.weights = weights