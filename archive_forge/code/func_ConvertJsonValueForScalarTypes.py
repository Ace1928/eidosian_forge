from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from collections import OrderedDict
import re
from apitools.base.py import extra_types
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import zip
def ConvertJsonValueForScalarTypes(scalar_type, scalar_value):
    """Convert the user input scalar value to JSON value.

  Args:
    scalar_type: String, the scalar type of the column, e.g INT64, DATE.
    scalar_value: String, the value of the column that user inputs.

  Returns:
    An API accepts JSON value of a column or an element of an array column.
  """
    if scalar_value == 'NULL':
        return extra_types.JsonValue(is_null=True)
    elif scalar_type == 'BOOL':
        bool_value = scalar_value.upper() == 'TRUE'
        return extra_types.JsonValue(boolean_value=bool_value)
    elif scalar_type == 'FLOAT64':
        if scalar_value in ('NaN', 'Infinity', '-Infinity'):
            return extra_types.JsonValue(string_value=scalar_value)
        else:
            return extra_types.JsonValue(double_value=float(scalar_value))
    else:
        return extra_types.JsonValue(string_value=scalar_value)