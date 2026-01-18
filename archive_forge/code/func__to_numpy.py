import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _to_numpy(a):
    """Converts Tensors, EagerTensors, and IndexedSlicesValue to numpy arrays.

  Args:
    a: any value.

  Returns:
    If a is EagerTensor or Tensor, returns the evaluation of a by calling
    numpy() or run(). If a is IndexedSlicesValue, constructs the corresponding
    dense numpy array. Otherwise returns a unchanged.
  """
    if isinstance(a, ops.EagerTensor):
        return a.numpy()
    if isinstance(a, tensor.Tensor):
        sess = ops.get_default_session()
        return sess.run(a)
    if isinstance(a, indexed_slices.IndexedSlicesValue):
        arr = np.zeros(a.dense_shape)
        assert len(a.values) == len(a.indices), 'IndexedSlicesValue has %s value slices but %s indices\n%s' % (a.values, a.indices, a)
        for values_slice, index in zip(a.values, a.indices):
            assert 0 <= index < len(arr), 'IndexedSlicesValue has invalid index %s\n%s' % (index, a)
            arr[index] += values_slice
        return arr
    return a