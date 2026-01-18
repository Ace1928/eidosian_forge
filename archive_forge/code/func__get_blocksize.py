from warnings import warn
import numpy as np
from ._matrix import spmatrix, _array_doc_to_matrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
def _get_blocksize(self):
    return self.data.shape[1:]