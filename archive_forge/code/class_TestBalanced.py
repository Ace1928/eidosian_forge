import numpy as np
from numpy.testing import assert_equal, assert_raises
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.grouputils import GroupSorted
class TestBalanced(CheckPanelLagMixin):

    @classmethod
    def setup_class(cls):
        cls.gind = np.repeat([0, 1, 2], 5)
        cls.mlag = 5
        x = np.arange(15)
        x += 10 ** cls.gind
        cls.x = x[:, None]
        cls.alle = {0: np.array([[1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 110, 111, 112, 113, 114]]), 1: np.array([[2, 3, 4, 5, 16, 17, 18, 19, 111, 112, 113, 114]]), 2: np.array([[3, 4, 5, 17, 18, 19, 112, 113, 114]]), 3: np.array([[4, 5, 18, 19, 113, 114]]), 4: np.array([[5, 19, 114]])}
        cls.calculate()