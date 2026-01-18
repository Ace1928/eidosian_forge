import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.tools import pca
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
provides a summary for the selection of the number of factors

        Returns
        -------
        sumstr : str
            summary of the results for selecting the number of factors

        