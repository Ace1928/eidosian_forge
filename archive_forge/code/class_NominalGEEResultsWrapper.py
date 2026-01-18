from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
class NominalGEEResultsWrapper(GEEResultsWrapper):
    pass