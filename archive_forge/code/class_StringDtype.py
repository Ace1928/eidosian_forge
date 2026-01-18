from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import pa_version_under10p1
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc
from pandas.core.dtypes.base import (
from pandas.core.dtypes.common import (
from pandas.core import ops
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import (
from pandas.core.arrays.integer import (
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow", "pyarrow_numpy"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """
    name: ClassVar[str] = 'string'

    @property
    def na_value(self) -> libmissing.NAType | float:
        if self.storage == 'pyarrow_numpy':
            return np.nan
        else:
            return libmissing.NA
    _metadata = ('storage',)

    def __init__(self, storage=None) -> None:
        if storage is None:
            infer_string = get_option('future.infer_string')
            if infer_string:
                storage = 'pyarrow_numpy'
            else:
                storage = get_option('mode.string_storage')
        if storage not in {'python', 'pyarrow', 'pyarrow_numpy'}:
            raise ValueError(f"Storage must be 'python', 'pyarrow' or 'pyarrow_numpy'. Got {storage} instead.")
        if storage in ('pyarrow', 'pyarrow_numpy') and pa_version_under10p1:
            raise ImportError('pyarrow>=10.0.1 is required for PyArrow backed StringArray.')
        self.storage = storage

    @property
    def type(self) -> type[str]:
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string == 'string':
            return cls()
        elif string == 'string[python]':
            return cls(storage='python')
        elif string == 'string[pyarrow]':
            return cls(storage='pyarrow')
        elif string == 'string[pyarrow_numpy]':
            return cls(storage='pyarrow_numpy')
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    def construct_array_type(self) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.string_arrow import ArrowStringArray, ArrowStringArrayNumpySemantics
        if self.storage == 'python':
            return StringArray
        elif self.storage == 'pyarrow':
            return ArrowStringArray
        else:
            return ArrowStringArrayNumpySemantics

    def __from_arrow__(self, array: pyarrow.Array | pyarrow.ChunkedArray) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        if self.storage == 'pyarrow':
            from pandas.core.arrays.string_arrow import ArrowStringArray
            return ArrowStringArray(array)
        elif self.storage == 'pyarrow_numpy':
            from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
            return ArrowStringArrayNumpySemantics(array)
        else:
            import pyarrow
            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                chunks = array.chunks
            results = []
            for arr in chunks:
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=libmissing.NA)
                results.append(arr)
        if len(chunks) == 0:
            arr = np.array([], dtype=object)
        else:
            arr = np.concatenate(results)
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(new_string_array, arr, StringDtype(storage='python'))
        return new_string_array