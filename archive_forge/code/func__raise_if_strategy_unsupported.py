from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
def _raise_if_strategy_unsupported(self):
    if not strategy_supports_loss_scaling():
        strategy = distribute_lib.get_strategy()
        if isinstance(strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1, tpu_strategy.TPUStrategyV2)):
            raise ValueError('Loss scaling is not supported with TPUStrategy. Loss scaling is unnecessary with TPUs, since they support bfloat16 instead of float16 and bfloat16 does not require loss scaling. You should remove the use of the LossScaleOptimizer when TPUs are used.')
        else:
            raise ValueError('Loss scaling is not supported with the tf.distribute.Strategy: %s. Try using a different Strategy, e.g. a MirroredStrategy' % strategy.__class__.__name__)