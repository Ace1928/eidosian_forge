import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def _process_kwds(self, kwds, ix):
    kwds = kwds.copy()
    for k in kwds:
        v = kwds[k]
        if isinstance(v, PatsyFormula):
            mat = patsy.dmatrix(v.formula, self.data, return_type='dataframe')
            mat = np.require(mat, requirements='W')[ix, :]
            if mat.shape[1] == 1:
                mat = mat[:, 0]
            kwds[k] = mat
    return kwds