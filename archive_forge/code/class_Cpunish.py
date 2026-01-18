import os
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from statsmodels.genmod.tests.results import glm_test_resids
class Cpunish:
    """
    The following are from the R script in models.datasets.cpunish
    Slightly different than published results, but should be correct
    Probably due to rounding in cleaning?
    """

    def __init__(self):
        self.params = (0.0002611017, 0.07781801, -0.09493111, 0.2969349, 2.301183, -18.72207, -6.80148)
        self.bse = (5.187132e-05, 0.07940193, 0.02291926, 0.4375164, 0.4283826, 4.283961, 4.14685)
        self.null_deviance = 136.57281747225
        self.df_null = 16
        self.deviance = 18.591641759528944
        self.df_resid = 10
        self.df_model = 6
        self.aic_R = 77.8546573896503
        self.aic_Stata = 4.579685683305706
        self.bic_Stata = -9.740492454486446
        self.chi2 = 128.8021169250578
        self.llf = -31.92732869482515
        self.scale = 1
        self.pearson_chi2 = 24.75374835
        self.resids = glm_test_resids.cpunish_resids
        self.fittedvalues = np.array([35.2263655, 8.1965744, 1.3118966, 3.6862982, 2.0823003, 1.0650316, 1.9260424, 2.4171405, 1.8473219, 2.8643241, 3.1211989, 3.3382067, 2.5269969, 0.8972542, 0.9793332, 0.5346209, 1.9790936])