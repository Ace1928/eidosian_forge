from __future__ import absolute_import
import re
def ToUpperCaseJsonString(value):
    """Coerces a primitive value into a upper-case JSON-compatible string.

  Special handling for  values whose JSON version is in upper-case.

  Args:
    value: value to convert.

  Returns:
    Value as a string.

  Raises:
    ValueError: when a non-primitive value is provided.
  """
    return str(value).upper()