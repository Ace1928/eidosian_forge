import numpy as np
from numpy.testing import assert_allclose
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.stats._diagnostic_other as diao
import statsmodels.discrete._diagnostics_count as diac
from statsmodels.base._parameter_inference import score_test
class TestScoreTestPoissonDispersed(TestScoreTestPoisson):
    rtol_ws = 0.11
    atol_ws = 0.015
    rtol_wooldridge = 0.03
    dispersed = True
    res_pvalue = [5.412978775609189e-14, 0.05027602575743518]
    res_disptest = np.array([[126.47363371056005, 0.0], [126.47363371056124, 0.0], [119.39362149777617, 0.0], [4.539405186430032, 5.641313974658654e-06], [4.539405186430032, 5.641313974658654e-06], [2.9164548934767525, 0.003540339101354978], [4.271414111277153, 1.9423733575592056e-05]])
    res_disptest_g = [17.670784788586968, 2.6262956791721383e-05]