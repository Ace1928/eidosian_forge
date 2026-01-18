from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.regression.linear_model import OLS
Example for gam.AdditiveModel and PolynomialSmoother

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.

Created on Fri Nov 04 13:45:43 2011

Author: Josef Perktold

