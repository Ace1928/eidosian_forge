import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils_v1 as dist_utils
from tensorflow.python.keras.engine import partial_batch_padding_handler as padding_util
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging
def _step_fn(ctx, inputs):
    """A step fn that returns update ops."""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        inputs, targets = inputs
    else:
        targets = None
    if isinstance(inputs, dict):
        inputs = [inputs[input_name] for input_name in model._feed_input_names]
    _build_model(strategy, model, mode, inputs, targets)
    grouped_inputs, grouped_outputs, grouped_updates, grouped_session_args = strategy.extended.call_for_each_replica(_per_replica_execution_function, args=(dist_utils.get_distributed_model(model, mode), mode))
    all_inputs, all_outputs, all_updates, all_session_args = dist_utils.unwrap_values(strategy, grouped_inputs, grouped_outputs, grouped_updates, grouped_session_args)
    combined_fn = backend.function(all_inputs, all_outputs, updates=all_updates, name='distributed_' + str(mode) + '_function', **all_session_args)
    for label, output in zip(output_labels, combined_fn.outputs):
        if label == 'loss':
            reduce_op = ds_reduce_util.ReduceOp.SUM
        else:
            reduce_op = ds_reduce_util.ReduceOp.MEAN
        ctx.set_last_step_output(label, output, reduce_op)
    return combined_fn.updates_op