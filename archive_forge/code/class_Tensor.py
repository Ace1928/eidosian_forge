from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@tf_export('Tensor', 'experimental.numpy.ndarray', v1=['Tensor'])
class Tensor(internal.NativeObject, core_tf_types.Symbol):
    """A `tf.Tensor` represents a multidimensional array of elements.

  All elements are of a single known data type.

  When writing a TensorFlow program, the main object that is
  manipulated and passed around is the `tf.Tensor`.

  A `tf.Tensor` has the following properties:

  * a single data type (float32, int32, or string, for example)
  * a shape

  TensorFlow supports eager execution and graph execution.  In eager
  execution, operations are evaluated immediately.  In graph
  execution, a computational graph is constructed for later
  evaluation.

  TensorFlow defaults to eager execution.  In the example below, the
  matrix multiplication results are calculated immediately.

  >>> # Compute some values using a Tensor
  >>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  >>> d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  >>> e = tf.matmul(c, d)
  >>> print(e)
  tf.Tensor(
  [[1. 3.]
   [3. 7.]], shape=(2, 2), dtype=float32)

  Note that during eager execution, you may discover your `Tensors` are actually
  of type `EagerTensor`.  This is an internal detail, but it does give you
  access to a useful function, `numpy`:

  >>> type(e)
  <class '...ops.EagerTensor'>
  >>> print(e.numpy())
    [[1. 3.]
     [3. 7.]]

  In TensorFlow, `tf.function`s are a common way to define graph execution.

  A Tensor's shape (that is, the rank of the Tensor and the size of
  each dimension) may not always be fully known.  In `tf.function`
  definitions, the shape may only be partially known.

  Most operations produce tensors of fully-known shapes if the shapes of their
  inputs are also fully known, but in some cases it's only possible to find the
  shape of a tensor at execution time.

  A number of specialized tensors are available: see `tf.Variable`,
  `tf.constant`, `tf.placeholder`, `tf.sparse.SparseTensor`, and
  `tf.RaggedTensor`.

  Caution: when constructing a tensor from a numpy array or pandas dataframe
  the underlying buffer may be re-used:

  ```python
  a = np.array([1, 2, 3])
  b = tf.constant(a)
  a[0] = 4
  print(b)  # tf.Tensor([4 2 3], shape=(3,), dtype=int64)
  ```

  Note: this is an implementation detail that is subject to change and users
  should not rely on this behaviour.

  For more on Tensors, see the [guide](https://tensorflow.org/guide/tensor).
  """
    OVERLOADABLE_OPERATORS = {'__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__div__', '__rdiv__', '__truediv__', '__rtruediv__', '__floordiv__', '__rfloordiv__', '__mod__', '__rmod__', '__lt__', '__le__', '__gt__', '__ge__', '__ne__', '__eq__', '__and__', '__rand__', '__or__', '__ror__', '__xor__', '__rxor__', '__getitem__', '__pow__', '__rpow__', '__invert__', '__neg__', '__abs__', '__matmul__', '__rmatmul__'}
    _USE_EQUALITY = tf2.enabled()

    def __getattr__(self, name):
        if name in {'T', 'astype', 'ravel', 'transpose', 'reshape', 'clip', 'size', 'tolist', 'data'}:
            raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'. " + '\n        If you are looking for numpy-related methods, please run the following:\n        tf.experimental.numpy.experimental_enable_numpy_behavior()\n      ')
        self.__getattribute__(name)

    @property
    def dtype(self):
        """The `DType` of elements in this tensor."""
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        """Returns a `tf.TensorShape` that represents the shape of this tensor.

    >>> t = tf.constant([1,2,3,4,5])
    >>> t.shape
    TensorShape([5])

    `tf.Tensor.shape` is equivalent to `tf.Tensor.get_shape()`.

    In a `tf.function` or when building a model using
    `tf.keras.Input`, they return the build-time shape of the
    tensor, which may be partially unknown.

    A `tf.TensorShape` is not a tensor. Use `tf.shape(t)` to get a tensor
    containing the shape, calculated at runtime.

    See `tf.Tensor.get_shape()`, and `tf.TensorShape` for details and examples.
    """
        if self._shape_val is None:
            dims, unknown_shape = self._shape
            if unknown_shape:
                self._shape_val = tensor_shape.unknown_shape()
            else:
                self._shape_val = tensor_shape.TensorShape(dims)
        return self._shape_val

    @property
    def ndim(self):
        return self.shape.rank

    def _disallow(self, task):
        raise errors.OperatorNotAllowedInGraphError(f'{task} is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.')

    def _disallow_bool_casting(self):
        self._disallow('Using a symbolic `tf.Tensor` as a Python `bool`')

    def _disallow_iteration(self):
        self._disallow('Iterating over a symbolic `tf.Tensor`')

    def __iter__(self):
        if not context.executing_eagerly():
            self._disallow_iteration()
        first_dim = self._get_first_dim()
        return _TensorIterator(self, first_dim)

    def _get_first_dim(self):
        shape = self._shape_tuple()
        if shape is None:
            raise TypeError('Cannot iterate over a tensor with unknown shape.')
        if not shape:
            raise TypeError('Cannot iterate over a scalar tensor.')
        if shape[0] is None:
            raise TypeError('Cannot iterate over a tensor with unknown first dimension.')
        return shape[0]

    def _shape_as_list(self):
        if self.shape.ndims is not None:
            return [dim.value for dim in self.shape.dims]
        else:
            return None

    def _shape_tuple(self):
        shape = self._shape_as_list()
        if shape is None:
            return None
        return tuple(shape)

    def _record_tape(self, capture):
        """Connect this graph tensor with capture for gradients calculation."""
        record.record_operation('captured_value', [self], [capture], backward_function=lambda x: [x], forward_function=lambda x: [x])

    def get_shape(self):
        """Returns a `tf.TensorShape` that represents the shape of this tensor.

    In eager execution the shape is always fully-known.

    >>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> print(a.shape)
    (2, 3)

    `tf.Tensor.get_shape()` is equivalent to `tf.Tensor.shape`.


    When executing in a `tf.function` or building a model using
    `tf.keras.Input`, `Tensor.shape` may return a partial shape (including
    `None` for unknown dimensions). See `tf.TensorShape` for more details.

    >>> inputs = tf.keras.Input(shape = [10])
    >>> # Unknown batch size
    >>> print(inputs.shape)
    (None, 10)

    The shape is computed using shape inference functions that are
    registered for each `tf.Operation`.

    The returned `tf.TensorShape` is determined at *build* time, without
    executing the underlying kernel. It is not a `tf.Tensor`. If you need a
    shape *tensor*, either convert the `tf.TensorShape` to a `tf.constant`, or
    use the `tf.shape(tensor)` function, which returns the tensor's shape at
    *execution* time.

    This is useful for debugging and providing early errors. For
    example, when tracing a `tf.function`, no ops are being executed, shapes
    may be unknown (See the [Concrete Functions
    Guide](https://www.tensorflow.org/guide/concrete_function) for details).

    >>> @tf.function
    ... def my_matmul(a, b):
    ...   result = a@b
    ...   # the `print` executes during tracing.
    ...   print("Result shape: ", result.shape)
    ...   return result

    The shape inference functions propagate shapes to the extent possible:

    >>> f = my_matmul.get_concrete_function(
    ...   tf.TensorSpec([None,3]),
    ...   tf.TensorSpec([3,5]))
    Result shape: (None, 5)

    Tracing may fail if a shape missmatch can be detected:

    >>> cf = my_matmul.get_concrete_function(
    ...   tf.TensorSpec([None,3]),
    ...   tf.TensorSpec([4,5]))
    Traceback (most recent call last):
    ...
    ValueError: Dimensions must be equal, but are 3 and 4 for 'matmul' (op:
    'MatMul') with input shapes: [?,3], [4,5].

    In some cases, the inferred shape may have unknown dimensions. If
    the caller has additional information about the values of these
    dimensions, `tf.ensure_shape` or `Tensor.set_shape()` can be used to augment
    the inferred shape.

    >>> @tf.function
    ... def my_fun(a):
    ...   a = tf.ensure_shape(a, [5, 5])
    ...   # the `print` executes during tracing.
    ...   print("Result shape: ", a.shape)
    ...   return a

    >>> cf = my_fun.get_concrete_function(
    ...   tf.TensorSpec([None, None]))
    Result shape: (5, 5)

    Returns:
      A `tf.TensorShape` representing the shape of this tensor.

    """
        return self.shape

    def set_shape(self, shape):
        """Updates the shape of this tensor.

    Note: It is recommended to use `tf.ensure_shape` instead of
    `Tensor.set_shape`, because `tf.ensure_shape` provides better checking for
    programming errors and can create guarantees for compiler
    optimization.

    With eager execution this operates as a shape assertion.
    Here the shapes match:

    >>> t = tf.constant([[1,2,3]])
    >>> t.set_shape([1, 3])

    Passing a `None` in the new shape allows any value for that axis:

    >>> t.set_shape([1,None])

    An error is raised if an incompatible shape is passed.

    >>> t.set_shape([1,5])
    Traceback (most recent call last):
    ...
    ValueError: Tensor's shape (1, 3) is not compatible with supplied
    shape [1, 5]

    When executing in a `tf.function`, or building a model using
    `tf.keras.Input`, `Tensor.set_shape` will *merge* the given `shape` with
    the current shape of this tensor, and set the tensor's shape to the
    merged value (see `tf.TensorShape.merge_with` for details):

    >>> t = tf.keras.Input(shape=[None, None, 3])
    >>> print(t.shape)
    (None, None, None, 3)

    Dimensions set to `None` are not updated:

    >>> t.set_shape([None, 224, 224, None])
    >>> print(t.shape)
    (None, 224, 224, 3)

    The main use case for this is to provide additional shape information
    that cannot be inferred from the graph alone.

    For example if you know all the images in a dataset have shape [28,28,3] you
    can set it with `tf.set_shape`:

    >>> @tf.function
    ... def load_image(filename):
    ...   raw = tf.io.read_file(filename)
    ...   image = tf.image.decode_png(raw, channels=3)
    ...   # the `print` executes during tracing.
    ...   print("Initial shape: ", image.shape)
    ...   image.set_shape([28, 28, 3])
    ...   print("Final shape: ", image.shape)
    ...   return image

    Trace the function, see the [Concrete Functions
    Guide](https://www.tensorflow.org/guide/concrete_function) for details.

    >>> cf = load_image.get_concrete_function(
    ...     tf.TensorSpec([], dtype=tf.string))
    Initial shape:  (None, None, 3)
    Final shape: (28, 28, 3)

    Similarly the `tf.io.parse_tensor` function could return a tensor with
    any shape, even the `tf.rank` is unknown. If you know that all your
    serialized tensors will be 2d, set it with `set_shape`:

    >>> @tf.function
    ... def my_parse(string_tensor):
    ...   result = tf.io.parse_tensor(string_tensor, out_type=tf.float32)
    ...   # the `print` executes during tracing.
    ...   print("Initial shape: ", result.shape)
    ...   result.set_shape([None, None])
    ...   print("Final shape: ", result.shape)
    ...   return result

    Trace the function

    >>> concrete_parse = my_parse.get_concrete_function(
    ...     tf.TensorSpec([], dtype=tf.string))
    Initial shape:  <unknown>
    Final shape:  (None, None)

    Make sure it works:

    >>> t = tf.ones([5,3], dtype=tf.float32)
    >>> serialized = tf.io.serialize_tensor(t)
    >>> print(serialized.dtype)
    <dtype: 'string'>
    >>> print(serialized.shape)
    ()
    >>> t2 = concrete_parse(serialized)
    >>> print(t2.shape)
    (5, 3)

    Caution: `set_shape` ensures that the applied shape is compatible with
    the existing shape, but it does not check at runtime. Setting
    incorrect shapes can result in inconsistencies between the
    statically-known graph and the runtime value of tensors. For runtime
    validation of the shape, use `tf.ensure_shape` instead. It also modifies
    the `shape` of the tensor.

    >>> # Serialize a rank-3 tensor
    >>> t = tf.ones([5,5,5], dtype=tf.float32)
    >>> serialized = tf.io.serialize_tensor(t)
    >>> # The function still runs, even though it `set_shape([None,None])`
    >>> t2 = concrete_parse(serialized)
    >>> print(t2.shape)
    (5, 5, 5)

    Args:
      shape: A `TensorShape` representing the shape of this tensor, a
        `TensorShapeProto`, a list, a tuple, or None.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
        self._shape_val = None
        if not isinstance(shape, tensor_shape.TensorShape):
            shape = tensor_shape.TensorShape(shape)
        dim_list = []
        if shape.dims is None:
            unknown_shape = True
        else:
            unknown_shape = False
            for dim in shape.dims:
                if dim.value is None:
                    dim_list.append(-1)
                else:
                    dim_list.append(dim.value)
        self._set_shape(dim_list, unknown_shape)

    def _as_node_def_input(self):
        """Return a value to use for the NodeDef "input" attribute.

    The returned string can be used in a NodeDef "input" attribute
    to indicate that the NodeDef uses this Tensor as input.

    Raises:
      ValueError: if this Tensor's Operation does not have a name.

    Returns:
      a string.
    """
        assert self._op.name
        if self.value_index == 0:
            return self._op.name
        else:
            return '%s:%d' % (self._op.name, self.value_index)

    def __str__(self):
        return 'Tensor("%s"%s%s%s)' % (self.name, ', shape=%s' % self.get_shape() if self.get_shape().ndims is not None else '', ', dtype=%s' % self._dtype.name if self._dtype else '', ', device=%s' % self.device if self.device else '')

    def __repr__(self):
        return "<tf.Tensor '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(), self._dtype.name)

    def __hash__(self):
        g = getattr(self, 'graph', None)
        if Tensor._USE_EQUALITY and (g is None or g.building_function):
            raise TypeError('Tensor is unhashable. Instead, use tensor.ref() as the key.')
        else:
            return id(self)
    __array_priority__ = 100

    def __array__(self, dtype=None):
        del dtype
        raise NotImplementedError(f"Cannot convert a symbolic tf.Tensor ({self.name}) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.")

    def __len__(self):
        raise TypeError(f'len is not well defined for a symbolic Tensor ({self.name}). Please call `x.shape` rather than `len(x)` for shape information.')

    @staticmethod
    def _override_operator(operator, func):
        _override_helper(Tensor, operator, func)

    def __bool__(self):
        """Dummy method to prevent a tensor from being used as a Python `bool`.

    This overload raises a `TypeError` when the user inadvertently
    treats a `Tensor` as a boolean (most commonly in an `if` or `while`
    statement), in code that was not converted by AutoGraph. For example:

    ```python
    if tf.constant(True):  # Will raise.
      # ...

    if tf.constant(5) < tf.constant(7):  # Will raise.
      # ...
    ```

    Raises:
      `TypeError`.
    """
        self._disallow_bool_casting()

    def __nonzero__(self):
        """Dummy method to prevent a tensor from being used as a Python `bool`.

    This is the Python 2.x counterpart to `__bool__()` above.

    Raises:
      `TypeError`.
    """
        self._disallow_bool_casting()

    def eval(self, feed_dict=None, session=None):
        """Evaluates this tensor in a `Session`.

    Note: If you are not using `compat.v1` libraries, you should not need this,
    (or `feed_dict` or `Session`).  In eager execution (or within `tf.function`)
    you do not need to call `eval`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.
    """
        return _eval_using_default_session(self, feed_dict, self.graph, session)

    @deprecation.deprecated(None, 'Use ref() instead.')
    def experimental_ref(self):
        return self.ref()

    def ref(self):
        """Returns a hashable reference object to this Tensor.

    The primary use case for this API is to put tensors in a set/dictionary.
    We can't put tensors in a set/dictionary as `tensor.__hash__()` is no longer
    available starting Tensorflow 2.0.

    The following will raise an exception starting 2.0

    >>> x = tf.constant(5)
    >>> y = tf.constant(10)
    >>> z = tf.constant(10)
    >>> tensor_set = {x, y, z}
    Traceback (most recent call last):
      ...
    TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
    >>> tensor_dict = {x: 'five', y: 'ten'}
    Traceback (most recent call last):
      ...
    TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.

    Instead, we can use `tensor.ref()`.

    >>> tensor_set = {x.ref(), y.ref(), z.ref()}
    >>> x.ref() in tensor_set
    True
    >>> tensor_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
    >>> tensor_dict[y.ref()]
    'ten'

    Also, the reference object provides `.deref()` function that returns the
    original Tensor.

    >>> x = tf.constant(5)
    >>> x.ref().deref()
    <tf.Tensor: shape=(), dtype=int32, numpy=5>
    """
        return object_identity.Reference(self)

    def __tf_tracing_type__(self, signature_context):
        if self.dtype == dtypes.resource or self.dtype == dtypes.variant:
            shape_inference_handle_data = handle_data_util.get_handle_data(self)
            handle_data = dtypes.HandleData(shape_inference_handle_data) if shape_inference_handle_data else None
            dtype = dtypes.DType(self.dtype._type_enum, handle_data)
        else:
            dtype = self.dtype
        spec = TensorSpec(self.shape, dtype)
        return spec

    def __tf_tensor__(self, dtype: Optional[dtypes.DType]=None, name: Optional[str]=None) -> 'Tensor':
        if dtype is not None and (not dtype.is_compatible_with(self.dtype)):
            raise ValueError(_add_error_prefix(f'Tensor conversion requested dtype {dtype.name} for Tensor with dtype {self.dtype.name}: {self!r}', name=name))
        return self