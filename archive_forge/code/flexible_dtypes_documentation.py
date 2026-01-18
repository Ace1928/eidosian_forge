import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import variables
from tensorflow.python.util import nest
Determine the result promotion dtype using the JNP-like promotion system.

  Args:
    *arrays_and_dtypes: A list of Tensors, Variables, NumPy arrays or python
      numbers.

  Returns:
    The result promotion type from all the inputs.
  