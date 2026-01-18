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
def SupportsAnsi(self):
    """Indicates whether the terminal appears to support ANSI escape sequences.

    Returns:
      bool: True if ANSI seems to be supported; False otherwise.
    """
    if console_attr_os.ForceEnableAnsi():
        return True
    return self._encoding != 'ascii' and ('screen' in self._term or 'xterm' in self._term)