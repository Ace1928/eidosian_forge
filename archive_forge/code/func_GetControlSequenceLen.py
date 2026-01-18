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
def GetControlSequenceLen(self, buf):
    """Returns the control sequence length at the beginning of buf.

    Used in display width computations. Control sequences have display width 0.

    Args:
      buf: The string to check for a control sequence.

    Returns:
      The conrol sequence length at the beginning of buf or 0 if buf does not
      start with a control sequence.
    """
    if not self._csi or not buf.startswith(self._csi):
        return 0
    n = 0
    for c in buf:
        n += 1
        if c.isalpha():
            break
    return n