import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class Cond(Operation):

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):

        def call_fn(*args, **kwargs):
            if not any_symbolic_tensors(args, kwargs):
                try:
                    return self.call(*args, **kwargs)
                except (TypeError, ValueError):
                    pass
            return self.symbolic_call(*args, **kwargs)
        if traceback_utils.is_traceback_filtering_enabled():
            call_fn = traceback_utils.inject_argument_info_in_traceback(call_fn, object_name=f'{self.__class__.__name__}.call()')
            return call_fn(*args, **kwargs)
        return call_fn(*args, **kwargs)

    def call(self, pred, true_fn, false_fn):
        return backend.core.cond(pred, true_fn, false_fn)

    def compute_output_spec(self, pred, true_fn, false_fn):

        def call_fn(fn):
            return fn()
        true_fn_spec = backend.compute_output_spec(call_fn, true_fn)
        false_fn_spec = backend.compute_output_spec(call_fn, false_fn)
        if not self._check_output_spec(true_fn_spec, false_fn_spec):
            raise ValueError(f'`true_fn` and `false_fn` should return outputs of the same kind (struct, dtype and shape). Got {true_fn_spec} and {false_fn_spec} instead.')
        return true_fn_spec

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