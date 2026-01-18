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
def _validate_csr_generation_inputs(num_rows, num_cols, density, distribution='uniform'):
    """Validates inputs for csr generation helper functions
    """
    total_nnz = int(num_rows * num_cols * density)
    if density < 0 or density > 1:
        raise ValueError('density has to be between 0 and 1')
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError('num_rows or num_cols should be greater than 0')
    if distribution == 'powerlaw':
        if total_nnz < 2 * num_rows:
            raise ValueError('not supported for this density: %s for this shape (%s, %s) Please keep : num_rows * num_cols * density >= 2 * num_rows' % (density, num_rows, num_cols))