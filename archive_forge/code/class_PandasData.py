from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
class PandasData(ModelData):
    """
    Data handling class which knows how to reattach pandas metadata to model
    results
    """

    def _convert_endog_exog(self, endog, exog=None):
        endog = np.asarray(endog)
        exog = exog if exog is None else np.asarray(exog)
        if endog.dtype == object or (exog is not None and exog.dtype == object):
            raise ValueError('Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).')
        return super()._convert_endog_exog(endog, exog)

    @classmethod
    def _drop_nans(cls, x, nan_mask):
        if isinstance(x, (Series, DataFrame)):
            return x.loc[nan_mask]
        else:
            return super()._drop_nans(x, nan_mask)

    @classmethod
    def _drop_nans_2d(cls, x, nan_mask):
        if isinstance(x, (Series, DataFrame)):
            return x.loc[nan_mask].loc[:, nan_mask]
        else:
            return super()._drop_nans_2d(x, nan_mask)

    def _check_integrity(self):
        endog, exog = (self.orig_endog, self.orig_exog)
        if exog is not None and (hasattr(endog, 'index') and hasattr(exog, 'index')) and (not self.orig_endog.index.equals(self.orig_exog.index)):
            raise ValueError('The indices for endog and exog are not aligned')
        super()._check_integrity()

    def _get_row_labels(self, arr):
        try:
            return arr.index
        except AttributeError:
            return self.orig_endog.index

    def attach_generic_columns(self, result, names):
        column_names = getattr(self, names, None)
        return Series(result, index=column_names)

    def attach_generic_columns_2d(self, result, rownames, colnames=None):
        colnames = colnames or rownames
        rownames = getattr(self, rownames, None)
        colnames = getattr(self, colnames, None)
        return DataFrame(result, index=rownames, columns=colnames)

    def attach_columns(self, result):
        if result.ndim <= 1:
            return Series(result, index=self.param_names)
        else:
            return DataFrame(result, index=self.param_names)

    def attach_columns_eq(self, result):
        return DataFrame(result, index=self.xnames, columns=self.ynames)

    def attach_cov(self, result):
        return DataFrame(result, index=self.cov_names, columns=self.cov_names)

    def attach_cov_eq(self, result):
        return DataFrame(result, index=self.ynames, columns=self.ynames)

    def attach_rows(self, result):
        squeezed = result.squeeze()
        k_endog = np.array(self.ynames, ndmin=1).shape[0]
        if k_endog > 1 and squeezed.shape == (k_endog,):
            squeezed = squeezed[None, :]
        if squeezed.ndim < 2:
            out = Series(squeezed)
        else:
            out = DataFrame(result)
            out.columns = self.ynames
        out.index = self.row_labels[-len(result):]
        return out

    def attach_dates(self, result):
        squeezed = result.squeeze()
        k_endog = np.array(self.ynames, ndmin=1).shape[0]
        if k_endog > 1 and squeezed.shape == (k_endog,):
            squeezed = np.asarray(squeezed)[None, :]
        if squeezed.ndim < 2:
            return Series(squeezed, index=self.predict_dates)
        else:
            return DataFrame(np.asarray(result), index=self.predict_dates, columns=self.ynames)

    def attach_mv_confint(self, result):
        return DataFrame(result.reshape((-1, 2)), index=self.cov_names, columns=['lower', 'upper'])

    def attach_ynames(self, result):
        squeezed = result.squeeze()
        if squeezed.ndim < 2:
            return Series(squeezed, name=self.ynames)
        else:
            return DataFrame(result, columns=self.ynames)