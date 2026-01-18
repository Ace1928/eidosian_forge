import numpy as np
from numpy.linalg import svd
import scipy
import pandas as pd
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from .multivariate_ols import multivariate_stats

            # Wilk's Chi square test of each canonical correlation
            df = (p - i + 1) * (q - i + 1)
            chi2 = a * np.log(prod)
            pval = stats.chi2.sf(chi2, df)
            stats.loc[i, 'Canonical correlation'] = self.cancorr[i]
            stats.loc[i, 'Chi-square'] = chi2
            stats.loc[i, 'DF'] = df
            stats.loc[i, 'Pr > ChiSq'] = pval
            