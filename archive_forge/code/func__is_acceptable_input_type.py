import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.util import nest
def _is_acceptable_input_type(x):
    """Determines if x is an acceptable input type for auto dtype conversion semantics."""
    supported_composite_types = (indexed_slices.IndexedSlices, weak_tensor.WeakTensor, variables.Variable)
    return isinstance(x, supported_composite_types) or not isinstance(x, composite_tensor.CompositeTensor)