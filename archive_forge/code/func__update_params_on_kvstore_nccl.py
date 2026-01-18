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
def _update_params_on_kvstore_nccl(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on NCCL kvstore."""
    valid_indices = [index for index, grad_list in enumerate(grad_arrays) if grad_list[0] is not None]
    valid_grad_arrays = [grad_arrays[i] for i in valid_indices]
    valid_param_arrays = [param_arrays[i] for i in valid_indices]
    valid_param_names = [param_names[i] for i in valid_indices]
    size = len(valid_grad_arrays)
    start = 0
    default_batch = '16'
    batch = int(os.getenv('MXNET_UPDATE_AGGREGATION_SIZE', default_batch))
    while start < size:
        end = start + batch if start + batch < size else size
        kvstore.pushpull(valid_param_names[start:end], valid_grad_arrays[start:end], out=valid_param_arrays[start:end], priority=-start)
        start = end