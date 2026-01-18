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
def _check_field_annotations(cls):
    """Validates the field annotations for tf.ExtensionType subclass `cls`."""
    annotations = getattr(cls, '__annotations__', {})
    for name, value in cls.__dict__.items():
        if name == 'Spec':
            if not isinstance(value, type):
                raise ValueError(f'{cls.__qualname__}.Spec must be a nested class; got {value}.')
            if value.__bases__ != (type_spec.TypeSpec,) and value.__bases__ != (object,):
                raise ValueError(f'{cls.__qualname__}.Spec must be directly subclassed from tf.TypeSpec.')
        elif extension_type_field.ExtensionTypeField.is_reserved_name(name):
            raise ValueError(f"The field annotations for {cls.__name__} are invalid. Field '{name}' is reserved.")
    for name in annotations:
        if extension_type_field.ExtensionTypeField.is_reserved_name(name):
            raise ValueError(f"The field annotations for {cls.__name__} are invalid. Field '{name}' is reserved.")
    for key, value in cls.__dict__.items():
        if not (key in annotations or callable(value) or key.startswith('_abc_') or (key == '_tf_extension_type_fields') or (key.startswith('__') and key.endswith('__')) or isinstance(value, (property, classmethod, staticmethod))):
            raise ValueError(f'The field annotations for {cls.__name__} are invalid. Field {key} is missing a type annotation.')