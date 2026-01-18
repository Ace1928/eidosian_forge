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
class _ExtensionTypeSpecCodec:
    """Codec for `tf.ExtensionTypeSpec`."""

    def can_encode(self, pyobj):
        """Returns true if `pyobj` can be encoded as an ExtensionTypeSpec."""
        if isinstance(pyobj, ExtensionTypeSpec):
            try:
                type_spec_registry.get_name(type(pyobj))
                return True
            except ValueError:
                return False
        return False

    def do_encode(self, extension_type_spec_value, encode_fn):
        """Returns an encoded proto for the given `tf.ExtensionTypeSpec`."""
        type_spec_class_name = type_spec_registry.get_name(type(extension_type_spec_value))
        type_state = extension_type_spec_value._serialize()
        num_flat_components = len(nest.flatten(extension_type_spec_value._component_specs, expand_composites=True))
        encoded_type_spec = struct_pb2.StructuredValue()
        encoded_type_spec.type_spec_value.CopyFrom(struct_pb2.TypeSpecProto(type_spec_class=struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC, type_state=encode_fn(type_state), type_spec_class_name=type_spec_class_name, num_flat_components=num_flat_components))
        return encoded_type_spec

    def can_decode(self, value):
        """Returns true if `value` can be decoded into a `tf.ExtensionTypeSpec`."""
        if value.HasField('type_spec_value'):
            type_spec_class_enum = value.type_spec_value.type_spec_class
            return type_spec_class_enum == struct_pb2.TypeSpecProto.EXTENSION_TYPE_SPEC
        return False

    def do_decode(self, value, decode_fn):
        """Returns the `tf.TypeSpec` encoded by the proto `value`."""
        type_spec_proto = value.type_spec_value
        class_name = type_spec_proto.type_spec_class_name
        try:
            type_spec_class = type_spec_registry.lookup(class_name)
        except ValueError:
            type_spec_class = AnonymousExtensionTypeSpec
            warnings.warn(f"The type '{class_name}' has not been registered. Falling back to using AnonymousExtensionTypeSpec instead.")
        return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))