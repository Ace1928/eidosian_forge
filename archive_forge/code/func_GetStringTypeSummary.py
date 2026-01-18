from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
import six
def GetStringTypeSummary(obj, available_space, line_length):
    """Returns a custom summary for string type objects.

  This function constructs a summary for string type objects by double quoting
  the string value. The double quoted string value will be potentially truncated
  with ellipsis depending on whether it has enough space available to show the
  full string value.

  Args:
    obj: The object to generate summary for.
    available_space: Number of character spaces available.
    line_length: The full width of the terminal, default is 80.

  Returns:
    A summary for the input object.
  """
    if len(obj) + len(TWO_DOUBLE_QUOTES) <= available_space:
        content = obj
    else:
        additional_len_needed = len(TWO_DOUBLE_QUOTES) + len(formatting.ELLIPSIS)
        if available_space < additional_len_needed:
            available_space = line_length
        content = formatting.EllipsisTruncate(obj, available_space - len(TWO_DOUBLE_QUOTES), line_length)
    return formatting.DoubleQuote(content)