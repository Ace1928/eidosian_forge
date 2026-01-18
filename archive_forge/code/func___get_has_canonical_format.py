import string
import warnings
import numpy
import cupy
import cupyx
from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _coo
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _index
def __get_has_canonical_format(self):
    """Determine whether the matrix has sorted indices and no duplicates.

        Returns
            bool: ``True`` if the above applies, otherwise ``False``.

        .. note::
            :attr:`has_canonical_format` implies :attr:`has_sorted_indices`, so
            if the latter flag is ``False``, so will the former be; if the
            former is found ``True``, the latter flag is also set.

        .. warning::
            Getting this property might synchronize the device.

        """
    if self.data.size == 0:
        self._has_canonical_format = True
    elif not getattr(self, '_has_sorted_indices', True):
        self._has_canonical_format = False
    elif not hasattr(self, '_has_canonical_format'):
        is_canonical = self._has_canonical_format_kern(self.indptr, self.indices, size=self.indptr.size - 1)
        self._has_canonical_format = bool(is_canonical.all())
    return self._has_canonical_format