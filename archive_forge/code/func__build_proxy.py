from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np
import pandas
from pandas._typing import IndexLabel
from pandas.core.dtypes.cast import find_common_type
from modin.error_message import ErrorMessage
@classmethod
def _build_proxy(cls, parent, column_name, materializer, dtype=None):
    """
        Construct a lazy proxy.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.
        materializer : callable(parent, column_name) -> pandas.CategoricalDtype
            A function to call in order to extract categorical values.
        dtype : dtype, optional
            The categories dtype.

        Returns
        -------
        LazyProxyCategoricalDtype
        """
    result = cls()
    result._parent = parent
    result._column_name = column_name
    result._materializer = materializer
    result._dtype = dtype
    return result