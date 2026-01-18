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
def _convert_fields(fields, field_values, context):
    """Type-checks and converts each field in `field_values` (in place).

  Args:
    fields: A list of `ExtensionTypeField` objects.
    field_values: A `dict` mapping field names to values.  Must contain an entry
      for each field.  I.e., `set(field_values.keys())` must be equal to
      `set([f.name for f in fields])`.
    context: _ConversionContext, indicates what kind of value we are converting.

  Raises:
    ValueError: If the keys of `field_values` do not match the names of
      the fields in `fields`.
    TypeError: If any value in `field_values` does not have the type indicated
      by the corresponding `ExtensionTypeField` object.
  """
    converted = {}
    if len(fields) != len(field_values):
        _report_field_mismatches(fields, field_values)
    for field in fields:
        if field.name not in field_values:
            _report_field_mismatches(fields, field_values)
        field_value = field_values[field.name]
        converted[field.name] = _convert_value(field_value, field.value_type, (field.name,), context)
    field_values.update(converted)