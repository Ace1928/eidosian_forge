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
@type_spec_registry.register('tf.AnonymousExtensionType.Spec')
class AnonymousExtensionTypeSpec(ExtensionTypeSpec):
    """TypeSpec for AnonymousExtensionType."""

    def __init__(self, **fields):
        for name in fields:
            if extension_type_field.ExtensionTypeField.is_reserved_name(name) or (name.startswith('__') and name.endswith('__')):
                raise ValueError(f'Reserved field name {name} was encountered when trying to instantiate an AnonymousExtensionTypeSpec.')
        fields = [(k, _convert_anonymous_fields(v, for_spec=True)) for k, v in fields.items()]
        self.__dict__.update(fields)
        super().__init__()
    value_type = AnonymousExtensionType

    def _serialize(self):
        return tuple(((name, _change_nested_mappings_to(value, dict)) for name, value in self.__dict__.items() if not extension_type_field.ExtensionTypeField.is_reserved_name(name)))

    def __setattr__(self, name, value):
        if name in type_spec.CACHED_FIXED_PROPERTIES:
            super().__setattr__(name, value)
        else:
            raise AttributeError(f'Cannot set attribute `{name}`. AnonymousExtensionTypeSpec instances are immutable.')

    def __delattr__(self, name):
        raise AttributeError(f'Cannot delete attribute `{name}`. AnonymousExtensionTypeSpec instances are immutable.')