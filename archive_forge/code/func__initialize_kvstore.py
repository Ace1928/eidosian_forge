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
def _initialize_kvstore(kvstore, param_arrays, arg_params, param_names, update_on_kvstore):
    """Initialize kvstore"""
    for idx, param_on_devs in enumerate(param_arrays):
        name = param_names[idx]
        if not update_on_kvstore or arg_params[name].stype != 'default':
            kvstore.init(name, arg_params[name])
        else:
            kvstore.broadcast(name, arg_params[name], out=param_on_devs)