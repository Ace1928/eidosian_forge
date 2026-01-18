from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
class TestTuckeyHSD4(CheckTuckeyHSDMixin):

    @classmethod
    def setup_class(cls):
        cls.endog = cylinders
        cls.groups = cyl_labels
        cls.alpha = 0.05
        cls.setup_class_()
        cls.res._simultaneous_ci()
        cls.halfwidth2 = np.array([1.5228335685980883, 0.9794949704444682, 0.7867380280553364, 2.3321237694566364, 0.5735513588275294])
        cls.meandiff2 = np.array([0.22222222222222232, 0.13333333333333375, 0.0, 2.2898550724637685, -0.08888888888888857, -0.22222222222222232, 2.067632850241546, -0.13333333333333375, 2.1565217391304348, 2.2898550724637685])
        cls.confint2 = np.array([-2.32022210717, 2.76466655161, -2.247517583, 2.51418424967, -3.66405224956, 3.66405224956, 0.113960166573, 4.46574997835, -1.87278583908, 1.6950080613, -3.529655688, 3.08521124356, 0.568180988881, 3.5670847116, -3.31822643175, 3.05155976508, 0.951206924521, 3.36183655374, -0.74487911754, 5.32458926247]).reshape(10, 2)
        cls.reject2 = np.array([False, False, False, True, False, False, True, False, True, False])

    def test_hochberg_intervals(self):
        assert_almost_equal(self.res.halfwidths, self.halfwidth2, 4)