from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
import six
def NeedsCustomDescription(component):
    """Whether the component should use a custom description and summary.

  Components of primitive type, such as ints, floats, dicts, lists, and others
  have messy builtin docstrings. These are inappropriate for display as
  descriptions and summaries in a CLI. This function determines whether the
  provided component has one of these docstrings.

  Note that an object such as `int` has the same docstring as an int like `3`.
  The docstring is OK for `int`, but is inappropriate as a docstring for `3`.

  Args:
    component: The component of interest.
  Returns:
    Whether the component should use a custom description and summary.
  """
    type_ = type(component)
    if type_ in six.string_types or type_ in six.integer_types or type_ is six.text_type or (type_ is six.binary_type) or (type_ in (float, complex, bool)) or (type_ in (dict, tuple, list, set, frozenset)):
        return True
    return False