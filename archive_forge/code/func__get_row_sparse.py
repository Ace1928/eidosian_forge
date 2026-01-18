from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _get_row_sparse(self, arr_list, ctx, row_id):
    """ Get row_sparse data from row_sparse parameters based on row_id. """
    if not isinstance(row_id, ndarray.NDArray):
        raise TypeError('row_id must have NDArray type, but %s is given' % type(row_id))
    if not self._trainer:
        raise RuntimeError("Cannot get row_sparse data for Parameter '%s' when no Trainer is created with it." % self.name)
    results = self._check_and_get(arr_list, ctx)
    self._trainer._row_sparse_pull(self, results, row_id)
    return results