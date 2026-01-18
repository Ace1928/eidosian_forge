from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
def ResetConsoleAttr(encoding=None):
    """Resets the console attribute state to the console default.

  Args:
    encoding: Reset to this encoding instead of the default.
      ascii -- ASCII. This is the default.
      utf8 -- UTF-8 unicode.
      win -- Windows code page 437.

  Returns:
    The global ConsoleAttr state object.
  """
    return GetConsoleAttr(encoding=encoding, reset=True)