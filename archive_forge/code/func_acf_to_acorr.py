from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def acf_to_acorr(acf):
    diag = np.diag(acf[0])
    return acf / np.sqrt(np.outer(diag, diag))