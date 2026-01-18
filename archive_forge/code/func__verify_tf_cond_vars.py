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
def _verify_tf_cond_vars(body_vars, orelse_vars, symbol_names):
    """Verifies variables manipulated by a conditional for consistency."""
    named_vars = zip(symbol_names, body_vars, orelse_vars)
    for name, body_var, orelse_var in named_vars:
        try:
            nest.assert_same_structure(body_var, orelse_var, expand_composites=True)
        except (ValueError, TypeError):
            try:
                body_var_tensors = variable_utils.convert_variables_to_tensors(body_var)
                orelse_var_tensors = variable_utils.convert_variables_to_tensors(orelse_var)
                nest.assert_same_structure(body_var_tensors, orelse_var_tensors, expand_composites=True)
            except (ValueError, TypeError) as e:
                raise TypeError("'{}' must have the same nested structure in the main and else branches:\n\n{}".format(name, str(e))) from e
        nest.map_structure(functools.partial(verify_single_cond_var, name), body_var, orelse_var)