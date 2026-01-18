import warnings
from collections import Counter
from itertools import chain
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import TransformerMixin, _fit_context, clone
from ..pipeline import _fit_transform_one, _name_estimators, _transform_one
from ..preprocessing import FunctionTransformer
from ..utils import Bunch, _get_column_indices, _safe_indexing
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._metadata_requests import METHODS
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._set_output import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _BaseComposition
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
class make_column_selector:
    """Create a callable to select columns to be used with
    :class:`ColumnTransformer`.

    :func:`make_column_selector` can select columns based on datatype or the
    columns name with a regex. When using multiple selection criteria, **all**
    criteria must match for a column to be selected.

    For an example of how to use :func:`make_column_selector` within a
    :class:`ColumnTransformer` to select columns based on data type (i.e.
    `dtype`), refer to
    :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py`.

    Parameters
    ----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If
        None, column selection will not be selected based on pattern.

    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    dtype_exclude : column dtype or list of column dtypes, default=None
        A selection of dtypes to exclude. For more details, see
        :meth:`pandas.DataFrame.select_dtypes`.

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.

    See Also
    --------
    ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.compose import make_column_selector
    >>> import numpy as np
    >>> import pandas as pd  # doctest: +SKIP
    >>> X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
    ...                   'rating': [5, 3, 4, 5]})  # doctest: +SKIP
    >>> ct = make_column_transformer(
    ...       (StandardScaler(),
    ...        make_column_selector(dtype_include=np.number)),  # rating
    ...       (OneHotEncoder(),
    ...        make_column_selector(dtype_include=object)))  # city
    >>> ct.fit_transform(X)  # doctest: +SKIP
    array([[ 0.90453403,  1.        ,  0.        ,  0.        ],
           [-1.50755672,  1.        ,  0.        ,  0.        ],
           [-0.30151134,  0.        ,  1.        ,  0.        ],
           [ 0.90453403,  0.        ,  0.        ,  1.        ]])
    """

    def __init__(self, pattern=None, *, dtype_include=None, dtype_exclude=None):
        self.pattern = pattern
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def __call__(self, df):
        """Callable for column selection to be used by a
        :class:`ColumnTransformer`.

        Parameters
        ----------
        df : dataframe of shape (n_features, n_samples)
            DataFrame to select columns from.
        """
        if not hasattr(df, 'iloc'):
            raise ValueError('make_column_selector can only be applied to pandas dataframes')
        df_row = df.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude)
        cols = df_row.columns
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]
        return cols.tolist()