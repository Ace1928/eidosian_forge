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
def _update_params_on_kvstore(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        name = param_names[index]
        if grad_list[0].stype == 'default' and arg_list[0].stype == 'default':
            kvstore.pushpull(name, grad_list, out=arg_list, priority=-index)
        else:
            kvstore.push(name, grad_list, priority=-index)
            kvstore.pull(name, out=arg_list, priority=-index)