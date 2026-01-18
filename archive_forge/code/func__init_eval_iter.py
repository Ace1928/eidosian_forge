import os
import time
import logging
import warnings
from collections import namedtuple
import numpy as np
from . import io
from . import ndarray as nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t
from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position
def _init_eval_iter(self, eval_data):
    """Initialize the iterator given eval_data."""
    if eval_data is None:
        return eval_data
    if isinstance(eval_data, (tuple, list)) and len(eval_data) == 2:
        if eval_data[0] is not None:
            if eval_data[1] is None and isinstance(eval_data[0], io.DataIter):
                return eval_data[0]
            input_data = np.array(eval_data[0]) if isinstance(eval_data[0], list) else eval_data[0]
            input_label = np.array(eval_data[1]) if isinstance(eval_data[1], list) else eval_data[1]
            return self._init_iter(input_data, input_label, is_train=True)
        else:
            raise ValueError('Eval data is NONE')
    if not isinstance(eval_data, io.DataIter):
        raise TypeError('Eval data must be DataIter, or NDArray/numpy.ndarray/list pair (i.e. tuple/list of length 2)')
    return eval_data