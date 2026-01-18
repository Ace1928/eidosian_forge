from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['layers.InputSpec'])
class InputSpec(object):
    """Specifies the rank, dtype and shape of every input to a layer.

  Layers can expose (if appropriate) an `input_spec` attribute:
  an instance of `InputSpec`, or a nested structure of `InputSpec` instances
  (one per input tensor). These objects enable the layer to run input
  compatibility checks for input structure, input rank, input shape, and
  input dtype.

  A None entry in a shape is compatible with any dimension,
  a None shape is compatible with any shape.

  Args:
    dtype: Expected DataType of the input.
    shape: Shape tuple, expected shape of the input
      (may include None for unchecked axes). Includes the batch size.
    ndim: Integer, expected rank of the input.
    max_ndim: Integer, maximum rank of the input.
    min_ndim: Integer, minimum rank of the input.
    axes: Dictionary mapping integer axes to
      a specific dimension value.
    allow_last_axis_squeeze: If True, then allow inputs of rank N+1 as long
      as the last axis of the input is 1, as well as inputs of rank N-1
      as long as the last axis of the spec is 1.
    name: Expected key corresponding to this input when passing data as
      a dictionary.

  Example:

  ```python
  class MyLayer(Layer):
      def __init__(self):
          super(MyLayer, self).__init__()
          # The layer will accept inputs with shape (?, 28, 28) & (?, 28, 28, 1)
          # and raise an appropriate error message otherwise.
          self.input_spec = InputSpec(
              shape=(None, 28, 28, 1),
              allow_last_axis_squeeze=True)
  ```
  """

    def __init__(self, dtype=None, shape=None, ndim=None, max_ndim=None, min_ndim=None, axes=None, allow_last_axis_squeeze=False, name=None):
        self.dtype = dtypes.as_dtype(dtype).name if dtype is not None else None
        shape = tensor_shape.TensorShape(shape)
        if shape.rank is None:
            shape = None
        else:
            shape = tuple(shape.as_list())
        if shape is not None:
            self.ndim = len(shape)
            self.shape = shape
        else:
            self.ndim = ndim
            self.shape = None
        self.max_ndim = max_ndim
        self.min_ndim = min_ndim
        self.name = name
        self.allow_last_axis_squeeze = allow_last_axis_squeeze
        try:
            axes = axes or {}
            self.axes = {int(k): axes[k] for k in axes}
        except (ValueError, TypeError):
            raise TypeError('The keys in axes must be integers.')
        if self.axes and (self.ndim is not None or self.max_ndim is not None):
            max_dim = (self.ndim if self.ndim else self.max_ndim) - 1
            max_axis = max(self.axes)
            if max_axis > max_dim:
                raise ValueError('Axis {} is greater than the maximum allowed value: {}'.format(max_axis, max_dim))

    def __repr__(self):
        spec = ['dtype=' + str(self.dtype) if self.dtype else '', 'shape=' + str(self.shape) if self.shape else '', 'ndim=' + str(self.ndim) if self.ndim else '', 'max_ndim=' + str(self.max_ndim) if self.max_ndim else '', 'min_ndim=' + str(self.min_ndim) if self.min_ndim else '', 'axes=' + str(self.axes) if self.axes else '']
        return 'InputSpec(%s)' % ', '.join((x for x in spec if x))

    def get_config(self):
        return {'dtype': self.dtype, 'shape': self.shape, 'ndim': self.ndim, 'max_ndim': self.max_ndim, 'min_ndim': self.min_ndim, 'axes': self.axes}

    @classmethod
    def from_config(cls, config):
        return cls(**config)