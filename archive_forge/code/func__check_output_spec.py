import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
def _check_output_spec(self, true_fn_spec, false_fn_spec):
    if true_fn_spec is None or false_fn_spec is None:
        return true_fn_spec is None and false_fn_spec is None
    elif isinstance(true_fn_spec, dict):
        if not isinstance(false_fn_spec, dict):
            return False
        if true_fn_spec.keys() != false_fn_spec.keys():
            return False
        if any((not self._check_output_spec(true_fn_spec[k], false_fn_spec[k]) for k in true_fn_spec.keys())):
            return False
    elif isinstance(true_fn_spec, list):
        if not isinstance(false_fn_spec, list):
            return False
        if len(true_fn_spec) != len(false_fn_spec):
            return False
        if any((not self._check_output_spec(ti, fi) for ti, fi in zip(true_fn_spec, false_fn_spec))):
            return False
    elif isinstance(true_fn_spec, tuple):
        if not isinstance(false_fn_spec, tuple):
            return False
        if len(true_fn_spec) != len(false_fn_spec):
            return False
        if any((not self._check_output_spec(ti, fi) for ti, fi in zip(true_fn_spec, false_fn_spec))):
            return False
    else:
        if true_fn_spec.dtype != false_fn_spec.dtype:
            return False
        if true_fn_spec.shape != false_fn_spec.shape:
            return False
    return True