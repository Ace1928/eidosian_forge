from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def _get_prediction_index(self, start, end, index=None, silent=False) -> tuple[int, int, int, Index | None]:
    """
        Get the location of a specific key in an index or model row labels

        Parameters
        ----------
        start : label
            The key at which to start prediction. Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        end : label
            The key at which to end prediction (note that this key will be
            *included* in prediction). Depending on the underlying
            model's index, may be an integer, a date (string, datetime object,
            pd.Timestamp, or pd.Period object), or some other object in the
            model's row labels.
        index : pd.Index, optional
            Optionally an index to associate the predicted results to. If None,
            an attempt is made to create an index for the predicted results
            from the model's index or model's row labels.
        silent : bool, optional
            Argument to silence warnings.

        Returns
        -------
        start : int
            The index / observation location at which to begin prediction.
        end : int
            The index / observation location at which to end in-sample
            prediction. The maximum value for this is nobs-1.
        out_of_sample : int
            The number of observations to forecast after the end of the sample.
        prediction_index : pd.Index or None
            The index associated with the prediction results. This index covers
            the range [start, end + out_of_sample]. If the model has no given
            index and no given row labels (i.e. endog/exog is not Pandas), then
            this will be None.

        Notes
        -----
        The arguments `start` and `end` behave differently, depending on if
        they are integer or not. If either is an integer, then it is assumed
        to refer to a *location* in the index, not to an index value. On the
        other hand, if it is a date string or some other type of object, then
        it is assumed to refer to an index *value*. In all cases, the returned
        `start` and `end` values refer to index *locations* (so in the former
        case, the given location is validated and returned whereas in the
        latter case a location is found that corresponds to the given index
        value).

        This difference in behavior is necessary to support `RangeIndex`. This
        is because integers for a RangeIndex could refer either to index values
        or to index locations in an ambiguous way (while for `NumericIndex`,
        since we have required them to be full indexes, there is no ambiguity).
        """
    nobs = len(self.endog)
    return get_prediction_index(start, end, nobs, base_index=self._index, index=index, silent=silent, index_none=self._index_none, index_generated=self._index_generated, data=self.data)