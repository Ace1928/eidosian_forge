from typing import Optional
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.types import core
class WeakTensor(extension_type.BatchableExtensionType, core.Tensor):
    """A weakly typed Tensor.

  A simple wrapper class that contains a normal Tensor.

  A "weak" type means that its dtype is temporarily inferred by the system,
  and could defer to other dtypes.

  i.g. weak f64 + f16 => f16

  This information is used for auto dtype conversion.
  """
    __name__ = 'tf.WeakTensor'
    tensor: tensor_lib.Tensor

    def __validate__(self):
        if self.tensor.dtype not in _ALLOWED_WEAK_DTYPES:
            raise TypeError(f'{self.tensor.dtype} not allowed as a weak type. The allowed types are {_ALLOWED_WEAK_DTYPES}.')

    def __str__(self):
        return self._format_weak_tensor(is_repr=False)

    def __repr__(self):
        return self._format_weak_tensor(is_repr=True)

    def _format_weak_tensor(self, is_repr):
        tensor_str = self.tensor.__repr__() if is_repr else self.tensor.__str__()
        closing_char = tensor_str[len(tensor_str) - 1]
        last_index = tensor_str.rfind(closing_char)
        return tensor_str[:last_index] + ', weak=True' + closing_char

    def __getattr__(self, *args, **kwargs):
        return getattr(self.tensor, *args, **kwargs)

    def _disallow(self, task):
        raise errors.OperatorNotAllowedInGraphError(f'{task} is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.')

    def _disallow_iteration(self):
        self._disallow('Iterating over a symbolic `tf.WeakTensor`')

    def _shape_as_list(self):
        if self.shape.ndims is not None:
            return [dim.value for dim in self.shape.dims]
        else:
            return None

    def __iter__(self):
        if not context.executing_eagerly():
            self._disallow_iteration()
        first_dim = self.tensor._get_first_dim()
        return _WeakTensorIterator(self, first_dim)

    def __hash__(self):
        return self.tensor.__hash__()

    def __copy__(self):
        return self

    def __len__(self):
        return self.tensor.__len__()

    def __bool__(self):
        return self.tensor.__bool__()

    def __tf_tensor__(self, dtype: Optional[dtypes.DType]=None, name: Optional[str]=None):
        return self.tensor.__tf_tensor__(dtype=dtype, name=name)

    def __deepcopy__(self, memo):
        del memo
        return self

    def to_tensor(self):
        """Converts this 'WeakTensor' into a 'tf.Tensor'."""
        return self.tensor

    def _as_graph_element(self):
        """Convert `self` to a graph element."""
        return self.tensor

    @classmethod
    def from_tensor(cls, tensor):
        """Converts a 'tf.Tensor' into a 'WeakTensor'.

    This should be the standard way of creating a WeakTensor instead
    of directly calling the WeakTensor constructor.

    Args:
      tensor: The `tf.Tensor` that should be converted into a 'WeakTensor'.

    Returns:
      A `EagerWeakTensor` or 'GraphWeakTensor' that holds the `tensor`.
    """
        if isinstance(tensor, core.Value):
            return EagerWeakTensor(tensor)
        if isinstance(tensor, core.Symbol):
            return GraphWeakTensor(tensor)
        raise errors.InvalidArgumentError(None, None, f'WeakTensor can only be constructed from tf.Tensor or tf.WeakTensor, but {type(tensor)} was given.')

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def is_tensor_like(self):
        return True
    __composite_gradient__ = WeakTensorGradient()