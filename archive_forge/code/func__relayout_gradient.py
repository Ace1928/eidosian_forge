import contextlib
import threading
from typing import Any, Callable, Optional, Sequence
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('Relayout')
def _relayout_gradient(op, grad):
    grad = gen_dtensor_ops.relayout_like(grad, layout_input=op.inputs[0])
    return grad