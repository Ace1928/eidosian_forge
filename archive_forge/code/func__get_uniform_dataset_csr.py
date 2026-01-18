import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def _get_uniform_dataset_csr(num_rows, num_cols, density=0.1, dtype=None, data_init=None, shuffle_csr_indices=False):
    """Returns CSRNDArray with uniform distribution
    This generates a csr matrix with totalnnz unique randomly chosen numbers
    from num_rows*num_cols and arranges them in the 2d array in the
    following way:
    row_index = (random_number_generated / num_rows)
    col_index = random_number_generated - row_index * num_cols
    """
    _validate_csr_generation_inputs(num_rows, num_cols, density, distribution='uniform')
    try:
        from scipy import sparse as spsp
        csr = spsp.rand(num_rows, num_cols, density, dtype=dtype, format='csr')
        if data_init is not None:
            csr.data.fill(data_init)
        if shuffle_csr_indices is True:
            shuffle_csr_column_indices(csr)
        result = mx.nd.sparse.csr_matrix((csr.data, csr.indices, csr.indptr), shape=(num_rows, num_cols), dtype=dtype)
    except ImportError:
        assert data_init is None, 'data_init option is not supported when scipy is absent'
        assert not shuffle_csr_indices, 'shuffle_csr_indices option is not supported when scipy is absent'
        dns = mx.nd.random.uniform(shape=(num_rows, num_cols), dtype=dtype)
        masked_dns = dns * (dns < density)
        result = masked_dns.tostype('csr')
    return result