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
def _py_while_stmt(test, body, get_state, set_state, opts):
    """Overload of while_stmt that executes a Python while loop."""
    del opts, get_state, set_state
    if __debug__:
        checker = _PythonLoopChecker()
        before_iteration = checker.before_iteration
        after_iteration = checker.after_iteration
        before_iteration()
        original_body = body

        def protected_body():
            original_body()
            after_iteration()
            before_iteration()
        body = protected_body

    def guarded_test():
        test_result = test()
        try:
            return bool(test_result)
        except errors_impl.OperatorNotAllowedInGraphError as e:
            ag_logging.log(1, 'Caught error while evaluating while loop condition', exc_info=True)
            raise NotImplementedError('The condition of while loop started as non-Tensor, then changed to Tensor. This may happen either because variables changed type, or when a break or return statement inside the loop depends on a Tensor condition. In both cases, changing to a TF loop should remove the error.\nSee https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#consistency-of-control-flow-types for more info.') from e
    while guarded_test():
        body()