from __future__ import absolute_import
import re
def StringToInt(handle_automatic=False):
    """Create conversion function which converts from a string to an integer.

  Args:
    handle_automatic: Boolean indicating whether a value of "automatic" should
      be converted to 0.

  Returns:
    A conversion function which converts a string to an integer.
  """

    def Convert(value):
        if value == 'automatic' and handle_automatic:
            return 0
        return int(value)
    return Convert