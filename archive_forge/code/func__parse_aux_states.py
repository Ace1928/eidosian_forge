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
def _parse_aux_states(sym, aux_states, ctx, dtype=default_dtype()):
    """Parses the given auxiliary states to a dictionary.

    Auxiliary states of the provided op `sym` are used as dictionary
    keys and elements of `aux_states` are used as values.

    Parameters
    ----------
    sym : Symbol
        Symbol containing op
    aux_states : None or list or dict
        Aux states

        - if type is list or tuple of `np.ndarray`
            inner elements are arrays correspoding to
            ``sym.list_auxiliary_states()``.
        - if type is dict of str -> `np.ndarray`
            maps the name of arguments to the corresponding `np.ndarray`.
        *In either case, all aux states of `sym` must be provided.*
    ctx : Context
        Device context.
    dtype: "asnumpy" or np.float16 or np.float32 or np.float64
        If dtype is "asnumpy" then the mx.nd.array created will have the same
        type as th numpy array from which it is copied.
        Otherwise, dtype is the explicit datatype for all mx.nd.array objects
        created in this function.

    Returns
    -------
    dict
        Dictionary with `sym` aux states as keys and `aux_states` elements
        as values.

    Examples
    -------
    >>> data = mx.symbol.Variable('data')
    >>> weight = mx.sym.Variable(name='fc1_weight')
    >>> fc1 = mx.symbol.FullyConnected(data = data, weight=weight, name='fc1', num_hidden=128)
    >>> fc2 = mx.symbol.BatchNorm(fc1, name='batchnorm0')
    >>> mean_states = np.ones(3)
    >>> var_states = np.ones(3)
    >>> _parse_aux_states(fc2, [mean_states, var_states], None)
    {'batchnorm0_moving_var': <NDArray 3 @cpu(0)>, 'batchnorm0_moving_mean': <NDArray 3 @cpu(0)>}
    >>> _parse_aux_states(fc2, {'batchnorm0_moving_var': mean_states,
    ...                         'batchnorm0_moving_mean': var_states}, None)
    {'batchnorm0_moving_var': <NDArray 3 @cpu(0)>, 'batchnorm0_moving_mean': <NDArray 3 @cpu(0)>}
    >>> _parse_aux_states(fc2, {'batchnorm0_moving_var': mean_states}, None)
    ValueError: Symbol aux_states names and given aux_states do not match.
    """
    assert dtype == 'asnumpy' or dtype in (np.float16, np.float32, np.float64)
    if aux_states is not None:
        if isinstance(aux_states, dict):
            if set(aux_states.keys()) != set(sym.list_auxiliary_states()):
                raise ValueError('Symbol aux_states names and given aux_states do not match.symbol aux_names:%s, aux_states.keys:%s' % (str(set(sym.list_auxiliary_states())), str(set(aux_states.keys()))))
        elif isinstance(aux_states, (list, tuple)):
            aux_names = sym.list_auxiliary_states()
            aux_states = {k: v for k, v in zip(aux_names, aux_states)}
        aux_states = {k: mx.nd.array(v, ctx=ctx, dtype=v.dtype if dtype == 'asnumpy' else dtype) for k, v in aux_states.items()}
    return aux_states