import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
class ExtensionTypeField(collections.namedtuple('ExtensionTypeField', ['name', 'value_type', 'default'])):
    """Metadata about a single field in a `tf.ExtensionType` object."""
    NO_DEFAULT = Sentinel('ExtensionTypeField.NO_DEFAULT')

    def __new__(cls, name, value_type, default=NO_DEFAULT):
        """Constructs a new ExtensionTypeField containing metadata for a single field.

    Args:
      name: The name of the new field (`str`).  May not be a reserved name.
      value_type: A python type expression constraining what values this field
        can take.
      default: The default value for the new field, or `NO_DEFAULT` if this
        field has no default value.

    Returns:
      A new `ExtensionTypeField`.

    Raises:
      TypeError: If the type described by `value_type` is not currently
          supported by `tf.ExtensionType`.
      TypeError: If `default` is specified and its type does not match
        `value_type`.
    """
        try:
            validate_field_value_type(value_type, allow_forward_references=True)
        except TypeError as e:
            raise TypeError(f'In field {name!r}: {e}') from e
        if default is not cls.NO_DEFAULT:
            default = _convert_value(default, value_type, (f'default value for {name}',), _ConversionContext.DEFAULT)
        return super(ExtensionTypeField, cls).__new__(cls, name, value_type, default)

    @staticmethod
    def is_reserved_name(name):
        """Returns true if `name` is a reserved name."""
        return name in RESERVED_FIELD_NAMES or name.lower().startswith('_tf_extension_type')