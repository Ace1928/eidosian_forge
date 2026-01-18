import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.ExtensionTypeBatchEncoder')
class ExtensionTypeBatchEncoder(type_spec.TypeSpecBatchEncoder):
    """Class used to encode and decode extension type values for batching.

  In order to be batched and unbatched by APIs such as `tf.data.Dataset`,
  `tf.keras`, and `tf.map_fn`, extension type values must be encoded as a list
  of `tf.Tensor`s, where stacking, unstacking, or concatenating these encoded
  tensors and then decoding the result must be equivalent to stacking,
  unstacking, or concatenating the original values. `ExtensionTypeBatchEncoder`s
  are responsible for implementing this encoding.

  The default `ExtensionTypeBatchEncoder` that is used by
  `BatchableExtensionType` assumes that extension type values can be stacked,
  unstacked, or concatenated by simply stacking, unstacking, or concatenating
  every nested `Tensor`, `ExtensionType`, `CompositeTensor`, and `TensorShape`
  field.

  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom encoder that overrides the `batch`,
  `unbatch`, `encode`, and `decode` methods. E.g.:

  >>> class CustomBatchEncoder(ExtensionTypeBatchEncoder):
  ...   pass # Override batch(), unbatch(), encode(), and decode().

  >>> class CustomType(BatchableExtensionType):
  ...   x: tf.Tensor
  ...   y: tf.Tensor
  ...   shape: tf.TensorShape
  ...   __batch_encoder__ = CustomBatchEncoder()

  For example, `tf.RaggedTensor` and `tf.SparseTensor` both use custom batch
  encodings which define ops to "box" and "unbox" individual values into
  `tf.variant` tensors.
  """

    def batch(self, spec, batch_size):
        """Returns the TypeSpec representing a batch of values described by `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that an outer axis with size `batch_size` is added to every nested
    `TypeSpec` and `TensorShape` field.  Subclasses may override this default
    definition, when necessary.

    Args:
      spec: The `TypeSpec` for an individual value.
      batch_size: An `int` indicating the number of values that are batched
        together, or `None` if the batch size is not known.

    Returns:
      A `TypeSpec` for a batch of values.
    """

        def batch_field(f):
            if isinstance(f, type_spec.BatchableTypeSpec):
                return f.__batch_encoder__.batch(f, batch_size)
            elif isinstance(f, tensor_shape.TensorShape):
                return [batch_size] + f
            else:
                return f
        fields = tuple(spec.__dict__.items())
        batched_fields = nest.map_structure(batch_field, fields)
        return _create_object_from_type_and_dict(type(spec), batched_fields)

    def unbatch(self, spec):
        """Returns the TypeSpec for a single unbatched element in `spec`.

    The default definition returns a `TypeSpec` that is equal to `spec`, except
    that the outermost axis is removed from every nested `TypeSpec`, and
    `TensorShape` field.  Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The `TypeSpec` for a batch of values.

    Returns:
      A `TypeSpec` for an individual value.
    """

        def unbatch_field(f):
            if isinstance(f, type_spec.BatchableTypeSpec):
                return f.__batch_encoder__.unbatch(f)
            elif isinstance(f, tensor_shape.TensorShape):
                return f[1:]
            else:
                return f
        fields = tuple(spec.__dict__.items())
        unbatched_fields = nest.map_structure(unbatch_field, fields)
        return _create_object_from_type_and_dict(type(spec), unbatched_fields)

    def encode(self, spec, value, minimum_rank=0):
        """Encodes `value` as a nest of batchable Tensors or CompositeTensors.

    The default definition returns a flat tuple of all the `Tensor`s,
    `CompositeTensor`s, and `ExtensionType`s from a depth-first traversal of
    `value`'s fields. Subclasses may override this default definition, when
    necessary.

    Args:
      spec: The TypeSpec of the value to encode.
      value: A value compatible with `spec`.
      minimum_rank: The minimum rank for the returned Tensors, CompositeTensors,
        and ExtensionType values.  This can be used to ensure that the encoded
        values can be unbatched this number of times.   If `minimum_rank>0`,
        then `t.shape[:minimum_rank]` must be compatible for all values `t`
        returned by `encode`.

    Returns:
      A nest (as defined by `tf.nest`) of `tf.Tensor`s, batchable
      `tf.CompositeTensor`s, or `tf.ExtensionType`s.  Stacking, unstacking, or
      concatenating these encoded values and then decoding the result must be
      equivalent to stacking, unstacking, or concatenating the original values.
    """
        return spec._to_components(value)

    def decode(self, spec, encoded_value):
        """Decodes `value` from a batchable tensor encoding.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec for the result value.  If encoded values with spec `s`
        were batched, then `spec` should be `s.batch(batch_size)`; or if encoded
        values with spec `s` were unbatched, then `spec` should be
        `s.unbatch()`.
      encoded_value: A nest of values returned by `encode`; or a nest of values
        that was formed by stacking, unstacking, or concatenating the
        corresponding elements of values returned by `encode`.

    Returns:
      A value compatible with `type_spec`.
    """
        return spec._from_components(encoded_value)

    def encoding_specs(self, spec):
        """Returns a list of `TensorSpec`(s) describing the encoding for `spec`.

    See `encode` for a description of the default encoding.  Subclasses may
    override this default definition, when necessary.

    Args:
      spec: The TypeSpec whose encoding should be described.

    Returns:
      A nest (as defined by `tf.nest) of `tf.TypeSpec`, describing the values
      that are returned by `self.encode(spec, ...)`.  All TypeSpecs in this
      nest must be batchable.
    """
        return spec._component_specs