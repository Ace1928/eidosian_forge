from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def GetRawKeyFunction():
    """Returns a function that reads one keypress from stdin with no echo.

  Returns:
    A function that reads one keypress from stdin with no echo or a function
    that always returns None if stdin does not support it.
  """
    for get_raw_key_function in (_GetRawKeyFunctionPosix, _GetRawKeyFunctionWindows):
        try:
            return get_raw_key_function()
        except:
            pass
    return lambda: None