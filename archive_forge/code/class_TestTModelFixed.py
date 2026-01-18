import pytest
import numpy as np
from numpy.testing import assert_allclose
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.miscmodels.tmodel import TLinearModel
class TestTModelFixed:

    @classmethod
    def setup_class(cls):
        endog = mm.m_marietta
        exog = add_constant(mm.CRSP)
        mod = TLinearModel(endog, exog, fix_df=3)
        res = mod.fit(method='bfgs', disp=False)
        modf = TLinearModel.from_formula('price ~ CRSP', data={'price': mm.m_marietta, 'CRSP': mm.CRSP}, fix_df=3)
        resf = modf.fit(method='bfgs', disp=False)
        cls.res1 = res
        cls.resf = resf
        cls.k_extra = 1

    @pytest.mark.smoke
    def test_smoke(self):
        res1 = self.res1
        resf = self.resf
        contr = np.eye(len(res1.params))
        res1.summary()
        res1.t_test(contr)
        res1.f_test(contr)
        resf.summary()
        resf.t_test(contr)
        resf.f_test(contr)