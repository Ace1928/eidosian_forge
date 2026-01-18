from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of the concentration parameter."""
    if not validate_args:
        return concentration
    concentration = distribution_util.embed_check_categorical_event_shape(concentration)
    return control_flow_ops.with_dependencies([check_ops.assert_positive(concentration, message='Concentration parameter must be positive.')], concentration)