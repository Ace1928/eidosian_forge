import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_arg_values
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
class ConvolutionOrthogonal3D(ConvolutionOrthogonal):
    """Initializer that generates a 3D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 5. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        if len(shape) != 5:
            raise ValueError(f'The tensor to initialize, specified by argument `shape` must be five-dimensional. Received shape={shape}')
        if shape[-2] > shape[-1]:
            raise ValueError(f'In_filters, specified by shape[-2]={shape[-2]} cannot be greater than out_filters, specified by shape[-1]={shape[-1]}.')
        if shape[0] != shape[1] or shape[0] != shape[2]:
            raise ValueError(f'Kernel sizes, specified by shape[0]={shape[0]},  shape[1]={shape[1]} and shape[2]={shape[2]} must be equal.')
        kernel = self._orthogonal_kernel(shape[0], shape[-2], shape[-1])
        kernel *= math_ops.cast(self.gain, dtype=dtype)
        return kernel

    def _dict_to_tensor(self, x, k1, k2, k3):
        """Convert a dictionary to a tensor.

    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.
      k3: Third dimension of x.

    Returns:
      A k1 * k2 * k3 tensor.
    """
        return array_ops_stack.stack([array_ops_stack.stack([array_ops_stack.stack([x[i, j, k] for k in range(k3)]) for j in range(k2)]) for i in range(k1)])

    def _block_orth(self, p1, p2, p3):
        """Construct a 3 x 3 kernel.

    Used to construct orthgonal kernel.

    Args:
      p1: A symmetric projection matrix.
      p2: A symmetric projection matrix.
      p3: A symmetric projection matrix.

    Returns:
      A 2 x 2 x 2 kernel.
    Raises:
      ValueError: If the dimensions of p1, p2 and p3 are different.
    """
        p1_shape = p1.shape.as_list()
        if p1_shape != p2.shape.as_list() or p1_shape != p3.shape.as_list():
            raise ValueError(f'The dimension of the matrices must be the same. Received p1.shape={p1.shape}, p2.shape={p2.shape} and p3.shape={p3.shape}.')
        n = p1_shape[0]
        eye = linalg_ops_impl.eye(n, dtype=self.dtype)
        kernel2x2x2 = {}

        def matmul(p1, p2, p3):
            return math_ops.matmul(math_ops.matmul(p1, p2), p3)

        def cast(i, p):
            """Return p or (1-p)."""
            return i * p + (1 - i) * (eye - p)
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    kernel2x2x2[i, j, k] = matmul(cast(i, p1), cast(j, p2), cast(k, p3))
        return kernel2x2x2

    def _matrix_conv(self, m1, m2):
        """Matrix convolution.

    Args:
      m1: is a k x k x k  dictionary, each element is a n x n matrix.
      m2: is a l x l x l dictionary, each element is a n x n matrix.

    Returns:
      (k + l - 1) x (k + l - 1) x (k + l - 1) dictionary each
      element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """
        n = m1[0, 0, 0].shape.as_list()[0]
        if n != m2[0, 0, 0].shape.as_list()[0]:
            raise ValueError(f'The entries in matrices m1 and m2 must have the same dimensions. Received m1[0, 0, 0].shape={m1[0, 0, 0].shape} and m2[0, 0, 0].shape={m2[0, 0, 0].shape}.')
        k = int(np.cbrt(len(m1)))
        l = int(np.cbrt(len(m2)))
        result = {}
        size = k + l - 1
        for i in range(size):
            for j in range(size):
                for r in range(size):
                    result[i, j, r] = array_ops.zeros([n, n], self.dtype)
                    for index1 in range(min(k, i + 1)):
                        for index2 in range(min(k, j + 1)):
                            for index3 in range(min(k, r + 1)):
                                if i - index1 < l and j - index2 < l and (r - index3 < l):
                                    result[i, j, r] += math_ops.matmul(m1[index1, index2, index3], m2[i - index1, j - index2, r - index3])
        return result

    def _orthogonal_kernel(self, ksize, cin, cout):
        """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
        if cin > cout:
            raise ValueError(f'The number of input channels (cin={cin}) cannot exceed the number of output channels (cout={cout}).')
        orth = self._orthogonal_matrix(cout)[0:cin, :]
        if ksize == 1:
            return array_ops.expand_dims(array_ops.expand_dims(array_ops.expand_dims(orth, 0), 0), 0)
        p = self._block_orth(self._symmetric_projection(cout), self._symmetric_projection(cout), self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout), self._symmetric_projection(cout), self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                for k in range(ksize):
                    p[i, j, k] = math_ops.matmul(orth, p[i, j, k])
        return self._dict_to_tensor(p, ksize, ksize, ksize)