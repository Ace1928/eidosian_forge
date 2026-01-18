import contextlib
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.mixed_precision import device_compatibility_check
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.experimental import mixed_precision_global_state
def _policy_equivalent_to_dtype(policy):
    """Returns True if the Policy is equivalent to a single dtype.

  A policy is equivalent to a single dtype if the policy's compute and variable
  dtypes are the same and the policy's type is Policy and not a subclass of
  Policy (such as PolicyV1).

  The "_infer" policy is considered equivalent to a single dtype.

  Args:
    policy: A Policy.

  Returns:
    True, if the policy is equivalent to a single dtype.
  """
    return type(policy) == Policy and list(policy.get_config().keys()) == ['name'] and (policy.name == '_infer' or _is_convertible_to_dtype(policy.name))