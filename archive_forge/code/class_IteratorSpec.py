import abc
import threading
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.IteratorSpec', v1=[])
class IteratorSpec(type_spec.TypeSpec):
    """Type specification for `tf.data.Iterator`.

  For instance, `tf.data.IteratorSpec` can be used to define a tf.function that
  takes `tf.data.Iterator` as an input argument:

  >>> @tf.function(input_signature=[tf.data.IteratorSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def square(iterator):
  ...   x = iterator.get_next()
  ...   return x * x
  >>> dataset = tf.data.Dataset.from_tensors(5)
  >>> iterator = iter(dataset)
  >>> print(square(iterator))
  tf.Tensor(25, shape=(), dtype=int32)

  Attributes:
    element_spec: A (nested) structure of `tf.TypeSpec` objects that represents
      the type specification of the iterator elements.
  """
    __slots__ = ['_element_spec']

    def __init__(self, element_spec):
        self._element_spec = element_spec

    @property
    def value_type(self):
        return OwnedIterator

    def _serialize(self):
        return (self._element_spec,)

    @property
    def _component_specs(self):
        return (tensor.TensorSpec([], dtypes.resource),)

    def _to_components(self, value):
        return (value._iterator_resource,)

    def _from_components(self, components):
        return OwnedIterator(dataset=None, components=components, element_spec=self._element_spec)

    @staticmethod
    def from_value(value):
        return IteratorSpec(value.element_spec)