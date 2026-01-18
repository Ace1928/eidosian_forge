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
def _init_predictor(self, input_shapes, type_dict=None):
    """Initialize the predictor module for running prediction."""
    shapes = {name: self.arg_params[name].shape for name in self.arg_params}
    shapes.update(dict(input_shapes))
    if self._pred_exec is not None:
        arg_shapes, _, _ = self.symbol.infer_shape(**shapes)
        assert arg_shapes is not None, 'Incomplete input shapes'
        pred_shapes = [x.shape for x in self._pred_exec.arg_arrays]
        if arg_shapes == pred_shapes:
            return
    pred_exec = self.symbol.simple_bind(self.ctx[0], grad_req='null', type_dict=type_dict, **shapes)
    pred_exec.copy_params_from(self.arg_params, self.aux_params)
    _check_arguments(self.symbol)
    self._pred_exec = pred_exec