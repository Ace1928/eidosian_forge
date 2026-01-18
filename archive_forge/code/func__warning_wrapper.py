from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def _warning_wrapper(*args, **kwargs):
    logging.warning('%s does not support dynamic loss scaling in symbolic and hybridized execution.', f.__name__)
    return f(*args, **kwargs)