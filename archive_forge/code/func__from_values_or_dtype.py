from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@classmethod
def _from_values_or_dtype(cls, values=None, categories=None, ordered: bool | None=None, dtype: Dtype | None=None) -> CategoricalDtype:
    """
        Construct dtype from the input parameters used in :class:`Categorical`.

        This constructor method specifically does not do the factorization
        step, if that is needed to find the categories. This constructor may
        therefore return ``CategoricalDtype(categories=None, ordered=None)``,
        which may not be useful. Additional steps may therefore have to be
        taken to create the final dtype.

        The return dtype is specified from the inputs in this prioritized
        order:
        1. if dtype is a CategoricalDtype, return dtype
        2. if dtype is the string 'category', create a CategoricalDtype from
           the supplied categories and ordered parameters, and return that.
        3. if values is a categorical, use value.dtype, but override it with
           categories and ordered if either/both of those are not None.
        4. if dtype is None and values is not a categorical, construct the
           dtype from categories and ordered, even if either of those is None.

        Parameters
        ----------
        values : list-like, optional
            The list-like must be 1-dimensional.
        categories : list-like, optional
            Categories for the CategoricalDtype.
        ordered : bool, optional
            Designating if the categories are ordered.
        dtype : CategoricalDtype or the string "category", optional
            If ``CategoricalDtype``, cannot be used together with
            `categories` or `ordered`.

        Returns
        -------
        CategoricalDtype

        Examples
        --------
        >>> pd.CategoricalDtype._from_values_or_dtype()
        CategoricalDtype(categories=None, ordered=None, categories_dtype=None)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     categories=['a', 'b'], ordered=True
        ... )
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        >>> dtype1 = pd.CategoricalDtype(['a', 'b'], ordered=True)
        >>> dtype2 = pd.CategoricalDtype(['x', 'y'], ordered=False)
        >>> c = pd.Categorical([0, 1], dtype=dtype1)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     c, ['x', 'y'], ordered=True, dtype=dtype2
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot specify `categories` or `ordered` together with
        `dtype`.

        The supplied dtype takes precedence over values' dtype:

        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)
        CategoricalDtype(categories=['x', 'y'], ordered=False, categories_dtype=object)
        """
    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == 'category':
                if ordered is None and cls.is_dtype(values):
                    ordered = values.dtype.ordered
                dtype = CategoricalDtype(categories, ordered)
            else:
                raise ValueError(f'Unknown dtype {repr(dtype)}')
        elif categories is not None or ordered is not None:
            raise ValueError('Cannot specify `categories` or `ordered` together with `dtype`.')
        elif not isinstance(dtype, CategoricalDtype):
            raise ValueError(f'Cannot not construct CategoricalDtype from {dtype}')
    elif cls.is_dtype(values):
        dtype = values.dtype._from_categorical_dtype(values.dtype, categories, ordered)
    else:
        dtype = CategoricalDtype(categories, ordered)
    return cast(CategoricalDtype, dtype)