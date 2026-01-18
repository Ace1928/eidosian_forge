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
@PublicAPI(stability='beta')
@pd.api.extensions.register_extension_dtype
class TensorDtype(pd.api.extensions.ExtensionDtype):
    """
    Pandas extension type for a column of homogeneous-typed tensors.

    This extension supports tensors in which the elements have different shapes.
    However, each tensor element must be non-ragged, i.e. each tensor element must have
    a well-defined, non-ragged shape.

    See:
    https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/base.py
    for up-to-date interface documentation and the subclassing contract. The
    docstrings of the below properties and methods were copied from the base
    ExtensionDtype.

    Examples:
        >>> # Create a DataFrame with a list of ndarrays as a column.
        >>> import pandas as pd
        >>> import numpy as np
        >>> import ray
        >>> df = pd.DataFrame({
        ...     "one": [1, 2, 3],
        ...     "two": list(np.arange(24).reshape((3, 2, 2, 2)))})
        >>> # Note the opaque np.object dtype for this column.
        >>> df.dtypes # doctest: +SKIP
        one     int64
        two    object
        dtype: object
        >>> # Cast column to our TensorDtype extension type.
        >>> from ray.data.extensions import TensorDtype
        >>> df["two"] = df["two"].astype(TensorDtype(np.int64, (3, 2, 2, 2)))
        >>> # Note that the column dtype is now TensorDtype instead of
        >>> # np.object.
        >>> df.dtypes # doctest: +SKIP
        one          int64
        two    TensorDtype(shape=(3, 2, 2, 2), dtype=int64)
        dtype: object
        >>> # Pandas is now aware of this tensor column, and we can do the
        >>> # typical DataFrame operations on this column.
        >>> col = 2 * (df["two"] + 10)
        >>> # The ndarrays underlying the tensor column will be manipulated,
        >>> # but the column itself will continue to be a Pandas type.
        >>> type(col) # doctest: +SKIP
        pandas.core.series.Series
        >>> col # doctest: +SKIP
        0   [[[ 2  4]
              [ 6  8]]
             [[10 12]
               [14 16]]]
        1   [[[18 20]
              [22 24]]
             [[26 28]
              [30 32]]]
        2   [[[34 36]
              [38 40]]
             [[42 44]
              [46 48]]]
        Name: two, dtype: TensorDtype(shape=(3, 2, 2, 2), dtype=int64)
        >>> # Once you do an aggregation on that column that returns a single
        >>> # row's value, you get back our TensorArrayElement type.
        >>> tensor = col.mean()
        >>> type(tensor) # doctest: +SKIP
        ray.data.extensions.tensor_extension.TensorArrayElement
        >>> tensor # doctest: +SKIP
        array([[[18., 20.],
                [22., 24.]],
               [[26., 28.],
                [30., 32.]]])
        >>> # This is a light wrapper around a NumPy ndarray, and can easily
        >>> # be converted to an ndarray.
        >>> type(tensor.to_numpy()) # doctest: +SKIP
        numpy.ndarray
        >>> # In addition to doing Pandas operations on the tensor column,
        >>> # you can now put the DataFrame into a Dataset.
        >>> ds = ray.data.from_pandas(df) # doctest: +SKIP
        >>> # Internally, this column is represented the corresponding
        >>> # Arrow tensor extension type.
        >>> ds.schema() # doctest: +SKIP
        one: int64
        two: extension<arrow.py_extension_type<ArrowTensorType>>
        >>> # You can write the dataset to Parquet.
        >>> ds.write_parquet("/some/path") # doctest: +SKIP
        >>> # And you can read it back.
        >>> read_ds = ray.data.read_parquet("/some/path") # doctest: +SKIP
        >>> read_ds.schema() # doctest: +SKIP
        one: int64
        two: extension<arrow.py_extension_type<ArrowTensorType>>
        >>> read_df = ray.get(read_ds.to_pandas_refs())[0] # doctest: +SKIP
        >>> read_df.dtypes # doctest: +SKIP
        one          int64
        two    TensorDtype(shape=(3, 2, 2, 2), dtype=int64)
        dtype: object
        >>> # The tensor extension type is preserved along the
        >>> # Pandas --> Arrow --> Parquet --> Arrow --> Pandas
        >>> # conversion chain.
        >>> read_df.equals(df) # doctest: +SKIP
        True
    """
    base = None

    def __init__(self, shape: Tuple[Optional[int], ...], dtype: np.dtype):
        self._shape = shape
        self._dtype = dtype

    @property
    def type(self):
        """
        The scalar type for the array, e.g. ``int``
        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``, assuming
        that value is valid (not NA). NA values do not need to be
        instances of `type`.
        """
        return TensorArrayElement

    @property
    def element_dtype(self):
        """
        The dtype of the underlying tensor elements.
        """
        return self._dtype

    @property
    def element_shape(self):
        """
        The shape of the underlying tensor elements. This will be a tuple of Nones if
        the corresponding TensorArray for this TensorDtype holds variable-shaped tensor
        elements.
        """
        return self._shape

    @property
    def is_variable_shaped(self):
        """
        Whether the corresponding TensorArray for this TensorDtype holds variable-shaped
        tensor elements.
        """
        return all((dim_size is None for dim_size in self.shape))

    @property
    def name(self) -> str:
        """
        A string identifying the data type.
        Will be used for display in, e.g. ``Series.dtype``
        """
        return f'numpy.ndarray(shape={self._shape}, dtype={self._dtype})'

    @classmethod
    def construct_from_string(cls, string: str):
        """
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[H]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\\[(?P<arg_name>.+)\\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        import ast
        import re
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        regex = '^(TensorDtype|numpy.ndarray)\\(shape=(\\((?:(?:\\d+|None),?\\s?)*\\)), dtype=(\\w+)\\)$'
        m = re.search(regex, string)
        err_msg = f"Cannot construct a '{cls.__name__}' from '{string}'; expected a string like 'TensorDtype(shape=(1, 2, 3), dtype=int64)'."
        if m is None:
            raise TypeError(err_msg)
        groups = m.groups()
        if len(groups) != 3:
            raise TypeError(err_msg)
        _, shape, dtype = groups
        shape = ast.literal_eval(shape)
        dtype = np.dtype(dtype)
        return cls(shape, dtype)

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return TensorArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        """
        Convert a pyarrow (chunked) array to a TensorArray.

        This and TensorArray.__arrow_array__ make up the
        Pandas extension type + array <--> Arrow extension type + array
        interoperability protocol. See
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#compatibility-with-apache-arrow
        for more information.
        """
        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks > 1:
                values = np.concatenate([chunk.to_numpy() for chunk in array.iterchunks()])
            else:
                values = array.chunk(0).to_numpy()
        else:
            values = array.to_numpy()
        return TensorArray(values)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    @property
    def _is_boolean(self):
        """
        Whether this extension array should be considered boolean.

        By default, ExtensionArrays are assumed to be non-numeric.
        Setting this to True will affect the behavior of several places,
        e.g.

        * is_bool
        * boolean indexing

        Returns
        -------
        bool
        """
        from pandas.core.dtypes.common import is_bool_dtype
        return is_bool_dtype(self._dtype)