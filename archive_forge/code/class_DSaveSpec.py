import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export
class DSaveSpec(saveable_object.SaveSpec):
    """DTensor SaveSpec that additionaly captures global_shape and layout."""

    def __init__(self, tensor, slice_spec, name, global_shape, layout, dtype=None, device=None):
        super().__init__(tensor=tensor, slice_spec=slice_spec, name=name, dtype=dtype, device=device)
        self.global_shape = global_shape
        self.layout = layout