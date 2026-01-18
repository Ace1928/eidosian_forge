from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _eager_cond_implementation(pred, true_fn, false_fn, strict, name):
    """Special cases for `cond` when executing eagerly."""
    pred = ops.convert_to_tensor(pred)
    pred_constant_value = tensor_util.constant_value(pred)
    if pred_constant_value is None:
        if not isinstance(true_fn, core.GenericFunction) or not isinstance(false_fn, core.GenericFunction):
            raise TypeError("When running tf.cond on a parallel device, 'true_fn' and 'false_fn' must be decorated with `tf.function`.")
        functions_run_eagerly = eager_function_run.functions_run_eagerly()
        if functions_run_eagerly:
            logging.warning('It looks like tf.function behavior was disabled, perhaps using tf.config.run_functions_eagerly. Parallelized tf.cond requires tf.function to work. This primitive will override the disable.')
        eager_function_run.run_functions_eagerly(False)
        try:
            return cond_v2.cond_v2(pred, true_fn, false_fn, name)
        finally:
            if functions_run_eagerly is not None:
                eager_function_run.run_functions_eagerly(functions_run_eagerly)
    else:
        with ops.name_scope(name, 'cond', [pred]):
            if pred_constant_value:
                result = true_fn()
            else:
                result = false_fn()
            if not strict:
                result = _UnpackIfSingleton(result)
            return result