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
class FakeOptimizerForRestoration(trackable.Trackable):
    """A fake optimizer used to support restoring TensorFlow 2.2 checkpoints.

  The checkpoint format for LossScaleOptimizers changed after TF 2.2. This class
  exists to support restoring TF 2.2 checkpoints in newer version of TensorFlow.

  In TF 2.2, LossScaleOptimizer would track the wrapped optimizer by calling the
  following in LossScaleOptimizer.__init__

  ```
  self._track_trackable(self._optimizer, 'base_optimizer')
  ```

  This means a dependency from the LossScaleOptimizer to the wrapped optimizer
  would be stored in the checkpoint. However now, the checkpoint format with a
  LossScaleOptimizer is the same as the format without a LossScaleOptimizer,
  except the loss scale is also stored. This means there is no dependency from
  the LossScaleOptimizer to the wrapped optimizer. Instead, the
  LossScaleOptimizer acts as if it is the wrapped optimizer, from a checkpoint's
  perspective, by overriding all Trackable methods and delegating them to the
  wrapped optimizer.

  To allow restoring TF 2.2. checkpoints, LossScaleOptimizer adds a dependency
  on this class instead of the inner optimizer. When restored, this class will
  instead restore the slot variables of the inner optimizer. Since this class
  has no variables, it does not affect the checkpoint when saved.
  """

    def __init__(self, optimizer):
        self._optimizer = optimizer

    def get_slot_names(self):
        return self._optimizer.get_slot_names()

    def _create_or_restore_slot_variable(self, slot_variable_position, slot_name, variable):
        return self._optimizer._create_or_restore_slot_variable(slot_variable_position, slot_name, variable)