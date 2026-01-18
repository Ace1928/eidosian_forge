import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
class TestFitResult:

    def test_plot_iv(self):
        rng = np.random.default_rng(1769658657308472721)
        data = stats.norm.rvs(0, 1, size=100, random_state=rng)

        def optimizer(*args, **kwargs):
            return differential_evolution(*args, **kwargs, seed=rng)
        bounds = [(0, 30), (0, 1)]
        res = stats.fit(stats.norm, data, bounds, optimizer=optimizer)
        try:
            import matplotlib
            message = "`plot_type` must be one of \\{'..."
            with pytest.raises(ValueError, match=message):
                res.plot(plot_type='llama')
        except (ModuleNotFoundError, ImportError):
            message = 'matplotlib must be installed to use method `plot`.'
            with pytest.raises(ModuleNotFoundError, match=message):
                res.plot(plot_type='llama')