from statsmodels.compat.python import lrange
import time
import numpy as np
from scipy import stats
from statsmodels.sandbox.gam import Model as GAM
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
Example for GAM with Poisson Model and PolynomialSmoother

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.

Created on Fri Nov 04 13:45:43 2011

Author: Josef Perktold
