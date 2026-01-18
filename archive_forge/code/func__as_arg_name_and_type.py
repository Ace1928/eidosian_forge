from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _as_arg_name_and_type(text):
    """Returns text as a name and type, if text looks like an arg name and type.

  Example:
    _as_arg_name_and_type("foo (int)") == "foo", "int"

  Args:
    text: The text, which may or may not be an arg name and type.
  Returns:
    The arg name and type, if text looks like an arg name and type.
    None otherwise.
  """
    tokens = text.split()
    if len(tokens) < 2:
        return None
    if _is_arg_name(tokens[0]):
        type_token = ' '.join(tokens[1:])
        type_token = type_token.lstrip('{([').rstrip('])}')
        return (tokens[0], type_token)
    else:
        return None