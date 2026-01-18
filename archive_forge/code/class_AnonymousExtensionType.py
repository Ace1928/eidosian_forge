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
class AnonymousExtensionType(ExtensionType):
    """Fallback used to decode `tf.ExtensionType` when the original type is unavailable.

  When a SavedModel is serialized, the signatures of any functions in the
  SavedModel can include `tf.ExtensionType` subclasses.  These subclasses are
  usually
  registered, so they can be restored when the SavedModel is loaded.  However,
  if a SavedModel is loaded without first registering the ExtensionType types in
  its
  signature, then the SavedModel will fall back to using the
  `AnonymousExtensionType`
  type instead.

  If necessary, `AnonymousExtensionType` objects can be converted to a concrete
  `tf.ExtensionType` subclass (and vice versa) using `reinterpret`.
  """
    _tf_extension_type_do_not_transform_this_class = True

    def __init__(self, **fields):
        for name in fields:
            if extension_type_field.ExtensionTypeField.is_reserved_name(name) or (name.startswith('__') and name.endswith('__')):
                raise ValueError(f'Reserved field name {name} was encountered when trying to instantiate an AnonymousExtensionType.')
        fields = [(k, _convert_anonymous_fields(v)) for k, v in fields.items()]
        self.__dict__.update(fields)
        self._tf_extension_type_convert_fields()
        super().__init__()

    @classmethod
    def _tf_extension_type_fields(cls):
        return [extension_type_field.ExtensionTypeField(name, None) for name in cls.__dict__ if not extension_type_field.ExtensionTypeField.is_reserved_name(name)]

    def __setattr__(self, name, value):
        raise AttributeError(f'Cannot set attribute `{name}`. AnonymousExtensionType instances are immutable.')

    def __delattr__(self, name):
        raise AttributeError(f'Cannot delete attribute `{name}`. AnonymousExtensionType instances are immutable.')

    def _tf_extension_type_convert_fields(self):
        fields = [(k, _convert_anonymous_fields(v)) for k, v in self.__dict__.items() if not extension_type_field.ExtensionTypeField.is_reserved_name(k)]
        self.__dict__.update(fields)

    def __repr__(self):
        fields = [f'{k}={v!r}' for k, v in self.__dict__.items() if not extension_type_field.ExtensionTypeField.is_reserved_name(k)]
        return f'AnonymousExtensionType({', '.join(fields)})'
    _tf_extension_type_cached_type_spec = None

    @property
    def _type_spec(self):
        if self._tf_extension_type_cached_type_spec is None:
            spec = AnonymousExtensionTypeSpec.from_value(self)
            self.__dict__['_tf_extension_type_cached_type_spec'] = spec
        return self._tf_extension_type_cached_type_spec