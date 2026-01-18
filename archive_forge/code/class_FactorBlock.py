from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
class FactorBlock(dict):
    """
    Helper class for describing and indexing a block of factors.

    Parameters
    ----------
    factor_names : tuple of str
        Tuple of factor names in the block (in the order that they will appear
        in the state vector).
    factor_order : int
        Order of the vector autoregression governing the factor block dynamics.
    endog_factor_map : pd.DataFrame
        Mapping from endog variable names to factor names.
    state_offset : int
        Offset of this factor block in the state vector.
    has_endog_Q : bool
        Flag if the model contains quarterly data.

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector that are associated with a
    particular block of factors.

    - `factors_ix` is a matrix of indices, with rows corresponding to factors
      in the block and columns corresponding to lags
    - `factors` is vec(factors_ix) (i.e. it stacks columns, so that it is
      `factors_ix.ravel(order='F')`). Thinking about a VAR system, the first
       k*p elements correspond to the equation for the first variable. The next
       k*p elements correspond to the equation for the second variable, and so
       on. It contains all of the lags in the state vector, which is max(5, p)
    - `factors_ar` is the subset of `factors` that have nonzero coefficients,
      so it contains lags up to p.
    - `factors_L1` only contains the first lag of the factors
    - `factors_L1_5` contains the first - fifth lags of the factors

    """

    def __init__(self, factor_names, factor_order, endog_factor_map, state_offset, k_endog_Q):
        self.factor_names = factor_names
        self.k_factors = len(self.factor_names)
        self.factor_order = factor_order
        self.endog_factor_map = endog_factor_map.loc[:, factor_names]
        self.state_offset = state_offset
        self.k_endog_Q = k_endog_Q
        if self.k_endog_Q > 0:
            self._factor_order = max(5, self.factor_order)
        else:
            self._factor_order = self.factor_order
        self.k_states = self.k_factors * self._factor_order
        self['factors'] = self.factors
        self['factors_ar'] = self.factors_ar
        self['factors_ix'] = self.factors_ix
        self['factors_L1'] = self.factors_L1
        self['factors_L1_5'] = self.factors_L1_5

    @property
    def factors_ix(self):
        """Factor state index array, shaped (k_factors, lags)."""
        o = self.state_offset
        return np.reshape(o + np.arange(self.k_factors * self._factor_order), (self._factor_order, self.k_factors)).T

    @property
    def factors(self):
        """Factors and all lags in the state vector (max(5, p))."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self._factor_order]

    @property
    def factors_ar(self):
        """Factors and all lags used in the factor autoregression (p)."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors * self.factor_order]

    @property
    def factors_L1(self):
        """Factors (first block / lag only)."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors]

    @property
    def factors_L1_5(self):
        """Factors plus four lags."""
        o = self.state_offset
        return np.s_[o:o + self.k_factors * 5]