import numpy as np
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .tools import (
from statsmodels.multivariate.pca import PCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
from statsmodels.compat.pandas import Appender

        Plot the coefficients of determination

        Parameters
        ----------
        endog_labels : bool, optional
            Whether or not to label the endogenous variables along the x-axis
            of the plots. Default is to include labels if there are 5 or fewer
            endogenous variables.
        fig : Figure, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the grid will be created in the provided
            figure using `fig.add_subplot()`.
        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----

        Produces a `k_factors` x 1 plot grid. The `i`th plot shows a bar plot
        of the coefficients of determination associated with factor `i`. The
        endogenous variables are arranged along the x-axis according to their
        position in the `endog` array.

        See Also
        --------
        coefficients_of_determination
        