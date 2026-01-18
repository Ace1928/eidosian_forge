from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
import six
Returns the predefined description for string obj.

  This function constructs a description for string type objects in the format
  of 'The string "<string_value>"'. <string_value> could be potentially
  truncated depending on whether it has enough space available to show the full
  string value.

  Args:
    obj: The object to generate description for.
    available_space: Number of character spaces available.
    line_length: The full width of the terminal, default if 80.

  Returns:
    A description for input object.
  