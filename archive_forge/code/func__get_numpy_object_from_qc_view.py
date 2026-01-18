import itertools
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.pandas.indexing import compute_sliced_len, is_range_like, is_slice, is_tuple
from modin.pandas.utils import is_scalar
from .arr import array
def _get_numpy_object_from_qc_view(self, qc_view, row_scalar: bool, col_scalar: bool, ndim: int):
    """
        Convert the query compiler view to the appropriate NumPy object.

        Parameters
        ----------
        qc_view : BaseQueryCompiler
            Query compiler to convert.
        row_scalar : bool
            Whether indexer for rows is scalar.
        col_scalar : bool
            Whether indexer for columns is scalar.
        ndim : {0, 1, 2}
            Number of dimensions in dataset to be retrieved.

        Returns
        -------
        modin.numpy.array
            The array object with the data from the query compiler view.

        Notes
        -----
        Usage of `slice(None)` as a lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
    if ndim == 2:
        return array(_query_compiler=qc_view, _ndim=self.arr._ndim)
    if self.arr._ndim == 1 and (not row_scalar):
        return array(_query_compiler=qc_view, _ndim=1)
    if self.arr._ndim == 1:
        _ndim = 0
    elif ndim == 0:
        _ndim = 0
    elif row_scalar and col_scalar:
        _ndim = 0
    elif not any([row_scalar, col_scalar]):
        _ndim = 2
    else:
        _ndim = 1
        if row_scalar:
            qc_view = qc_view.transpose()
    if _ndim == 0:
        return qc_view.to_numpy()[0, 0]
    res_arr = array(_query_compiler=qc_view, _ndim=_ndim)
    return res_arr