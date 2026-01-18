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
def _predict_step_fn(inputs):
    """A fn that returns output of single prediction step."""
    distribute_lib.get_replica_context().merge_call(_build_model, args=(model, mode, inputs))
    _, outputs, updates, _ = _per_replica_execution_function(dist_utils.get_distributed_model(model, mode), mode)
    with ops.control_dependencies([updates]):
        return [array_ops.identity(out) for out in outputs]