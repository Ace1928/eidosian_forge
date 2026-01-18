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
def GetRawKey(self):
    """Reads one key press from stdin with no echo.

    Returns:
      The key name, None for EOF, <KEY-*> for function keys, otherwise a
      character.
    """
    return self._get_raw_key[0]()