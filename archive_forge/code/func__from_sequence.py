import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
@classmethod
def _from_sequence(cls, scalars, *, dtype: Optional[Dtype]=None, copy: bool=False):
    """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this
            method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
    if copy and isinstance(scalars, np.ndarray):
        scalars = scalars.copy()
    elif isinstance(scalars, TensorArray):
        scalars = scalars._tensor.copy() if copy else scalars._tensor
    return TensorArray(scalars)