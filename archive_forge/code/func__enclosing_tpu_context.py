import collections
import functools
import re
import string
import numpy as np
import opt_einsum
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _enclosing_tpu_context():
    context = ops.get_default_graph()._get_control_flow_context()
    while context is not None and (not isinstance(context, control_flow_ops.XLAControlFlowContext)):
        context = context.outer_context
    return context