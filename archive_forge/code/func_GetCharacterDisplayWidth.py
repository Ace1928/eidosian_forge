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
def GetCharacterDisplayWidth(char):
    """Returns the monospaced terminal display width of char.

  Assumptions:
    - monospaced display
    - ambiguous or unknown chars default to width 1
    - ASCII control char width is 1 => don't use this for control chars

  Args:
    char: The character to determine the display width of.

  Returns:
    The monospaced terminal display width of char: either 0, 1, or 2.
  """
    if not isinstance(char, six.text_type):
        return 1
    char = unicodedata.normalize('NFC', char)
    if unicodedata.combining(char) != 0:
        return 0
    elif unicodedata.category(char) == 'Cf':
        return 0
    elif unicodedata.east_asian_width(char) in 'FW':
        return 2
    else:
        return 1