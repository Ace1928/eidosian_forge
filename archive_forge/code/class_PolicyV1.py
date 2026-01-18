import contextlib
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
class PolicyV1(Policy):
    """A deprecated dtype policy for a Keras layer.

  Warning: This class is now deprecated and will be removed soon. Please use the
  non-experimental class `tf.keras.mixed_precision.Policy` instead.

  The difference between this class and the non-experimental class is that this
  class has a `loss_scale` field and the non-experimental class does not. The
  loss scale is only used by `tf.keras.Model.compile`, which automatically wraps
  the optimizer with a `LossScaleOptimizer` if the optimizer is not already a
  `LossScaleOptimizer`. For the non-experimental Policy class, `Model.compile`
  instead wraps the optimizer with a `LossScaleOptimizer` if `Policy.name` is
  "mixed_float16".

  When deserializing objects with an experimental policy using functions like
  `tf.keras.utils.deserialize_keras_object`, the policy will be deserialized as
  the non-experimental `tf.keras.mixed_precision.Policy`, and the loss scale
  will silently be dropped. This is so that SavedModels that are generated
  with an experimental policy can be restored after the experimental policy is
  removed.
  """

    def __init__(self, name, loss_scale='auto'):
        """Constructs the policy.

    The `name` argument determines the compute and variable dtype, the default
    loss scale, and has no additional effect on the Policy. The compute and
    variable dtypes can only be specified through `name`, and cannot be
    specified directly.

    Args:
      name: A string. Can be one of the following values:
        * Any dtype name, such as 'float32' or 'float64'. Both the variable and
          compute dtypes will be that dtype.
        * 'mixed_float16' or 'mixed_bfloat16': The compute dtype is float16 or
          bfloat16, while the variable dtype is float32. With 'mixed_float16',
          a dynamic loss scale is used. These policies are used for mixed
          precision training.
      loss_scale: A `tf.compat.v1.mixed_precision.LossScale`, an int (which
        uses a `FixedLossScale`), the string "dynamic" (which uses a
        `DynamicLossScale`), or None (which uses no loss scale). Defaults to
        `"auto"`. In the `"auto"` case: 1) if `name` is `"mixed_float16"`, then
        use `loss_scale="dynamic"`. 2) otherwise, do not use a loss scale. Only
        `tf.keras.Model`s, not layers, use the loss scale, and it is only used
        during `Model.fit`, `Model.train_on_batch`, and other similar methods.
    """
        super(PolicyV1, self).__init__(name)
        if loss_scale == 'auto':
            loss_scale = 'dynamic' if name == 'mixed_float16' else None
            self._using_default_loss_scale = True
        else:
            self._using_default_loss_scale = False
        if loss_scale and self._compute_dtype not in (None, 'float16'):
            tf_logging.warning('Creating a Policy with a loss scale is only useful for float16 policies. You passed loss_scale=%r for policy %s. Consider not passing any loss_scale instead.' % (loss_scale, name))
        self._loss_scale = keras_loss_scale_module.get(loss_scale)

    @property
    def loss_scale(self):
        """Returns the loss scale of this Policy.

    Returns:
      A `tf.compat.v1.mixed_precision.experimental.LossScale`, or None.
    """
        return self._loss_scale

    def __repr__(self):
        return '<PolicyV1 "%s", loss_scale=%s>' % (self._name, self.loss_scale)

    def get_config(self):
        config = {'name': self.name}
        if not self._using_default_loss_scale:
            config['loss_scale'] = keras_loss_scale_module.serialize(self.loss_scale)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'loss_scale' in config and isinstance(config['loss_scale'], dict):
            config = config.copy()
            config['loss_scale'] = keras_loss_scale_module.deserialize(config['loss_scale'], custom_objects=custom_objects)
        return cls(**config)