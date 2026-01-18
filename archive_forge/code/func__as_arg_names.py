from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _as_arg_names(names_str):
    """Converts names_str to a list of arg names.

  Example:
    _as_arg_names("a, b, c") == ["a", "b", "c"]

  Args:
    names_str: A string with multiple space or comma separated arg names.
  Returns:
    A list of arg names, or None if names_str doesn't look like a list of arg
    names.
  """
    names = re.split(',| ', names_str)
    names = [name.strip() for name in names if name.strip()]
    for name in names:
        if not _is_arg_name(name):
            return None
    if not names:
        return None
    return names