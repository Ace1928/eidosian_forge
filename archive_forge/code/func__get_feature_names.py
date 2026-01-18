import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _get_feature_names(X):
    """Get feature names from X.

    Support for other array containers should place its implementation here.

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.

        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None
    if _is_pandas_df(X):
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, '__dataframe__'):
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)
    if feature_names is None or len(feature_names) == 0:
        return
    types = sorted((t.__qualname__ for t in set((type(v) for v in feature_names))))
    if len(types) > 1 and 'str' in types:
        raise TypeError(f'Feature names are only supported if all input features have string names, but your input has {types} as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.')
    if len(types) == 1 and types[0] == 'str':
        return feature_names