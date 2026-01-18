import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _verify_single_loop_var(name, check_shape, init, entry, exit_, shape_invariant):
    """Verifies whether the initial, entry and exit values are consistent."""
    assert entry is not None, "no TF op should set '{}' to None?".format(name)
    if exit_ is None:
        raise ValueError("'{}' is None at the end of the iteration.".format(name))
    if isinstance(init, (bool, int, float, str, np.ndarray)):
        init = tensor_conversion.convert_to_tensor_v2(init)
    if isinstance(entry, (bool, int, float, str, np.ndarray)):
        entry = tensor_conversion.convert_to_tensor_v2(entry)
    if isinstance(exit_, (bool, int, float, str, np.ndarray)):
        exit_ = tensor_conversion.convert_to_tensor_v2(exit_)
    if not tensor_util.is_tf_type(entry) or not tensor_util.is_tf_type(exit_):
        return
    if not hasattr(entry, 'dtype') or not hasattr(exit_, 'dtype'):
        return
    if not hasattr(entry, 'shape') or not hasattr(exit_, 'shape'):
        return
    if entry.dtype != exit_.dtype:
        raise TypeError("'{}' has dtype {} before the loop, but dtype {} after one iteration".format(name, entry.dtype.name, exit_.dtype.name))
    if check_shape:
        exit_shape = exit_.shape
        if shape_invariant is None:
            entry_shape = entry.shape
            if not _is_subshape(exit_shape, entry_shape):
                raise ValueError("'{}' has shape {} before the loop, but shape {} after one iteration. Use tf.autograph.experimental.set_loop_options to set shape invariants.".format(name, entry_shape, exit_shape))
        else:
            init_shape = init.shape
            if not _is_subshape(init_shape, shape_invariant):
                raise ValueError("'{}' has shape {} before the loop, which does not conform with the shape invariant {}.".format(name, init_shape, shape_invariant))
            if not _is_subshape(exit_shape, shape_invariant):
                raise ValueError("'{}' has shape {} after one iteration, which does not conform with the shape invariant {}.".format(name, exit_shape, shape_invariant))