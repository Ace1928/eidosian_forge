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
def DisplayWidth(self, buf):
    """Returns the display width of buf, handling unicode and ANSI controls.

    Args:
      buf: The string to count from.

    Returns:
      The display width of buf, handling unicode and ANSI controls.
    """
    if not isinstance(buf, six.string_types):
        return len(buf)
    cached = self._display_width_cache.get(buf, None)
    if cached is not None:
        return cached
    width = 0
    max_width = 0
    i = 0
    while i < len(buf):
        if self._csi and buf[i:].startswith(self._csi):
            i += self.GetControlSequenceLen(buf[i:])
        elif buf[i] == '\n':
            max_width = max(width, max_width)
            width = 0
            i += 1
        else:
            width += GetCharacterDisplayWidth(buf[i])
            i += 1
    max_width = max(width, max_width)
    self._display_width_cache[buf] = max_width
    return max_width