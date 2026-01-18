import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids
class Scotvote:
    """
    Scotvot class is used with TestGlmGamma.
    """

    def __init__(self):
        self.params = (4.961768e-05, 0.002034423, -7.181429e-05, 0.000111852, -1.467515e-07, -0.0005186831, -2.42717498e-06, -0.01776527)
        self.bse = (1.621577e-05, 0.0005320802, 2.711664e-05, 4.057691e-05, 1.236569e-07, 0.0002402534, 7.460253e-07, 0.01147922)
        self.null_deviance = 0.536072
        self.df_null = 31
        self.deviance = 0.087388516417
        self.df_resid = 24
        self.df_model = 7
        self.aic_R = 182.947045954721
        self.aic_Stata = 10.72212
        self.bic_Stata = -83.09027
        self.llf = -163.5539382
        self.scale = 0.003584283
        self.pearson_chi2 = 0.0860228056
        self.prsquared = 0.429
        self.prsquared_cox_snell = 0.97971
        self.resids = glm_test_resids.scotvote_resids
        self.fittedvalues = np.array([57.80431482, 53.2733447, 50.56347993, 58.33003783, 70.46562169, 56.88801284, 66.81878401, 66.03410393, 57.92937473, 63.23216907, 53.9914785, 61.28993391, 64.81036393, 63.47546816, 60.69696114, 74.83508176, 56.56991106, 72.01804172, 64.35676519, 52.02445881, 64.24933079, 71.15070332, 45.73479688, 54.93318588, 66.98031261, 52.02479973, 56.18413736, 58.12267471, 67.37947398, 60.49162862, 73.82609217, 69.61515621])