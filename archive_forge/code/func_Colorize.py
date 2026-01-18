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
def Colorize(self, string, color, justify=None):
    """Generates a colorized string, optionally justified.

    Args:
      string: The string to write.
      color: The color name -- must be in _ANSI_COLOR.
      justify: The justification function, no justification if None. For
        example, justify=lambda s: s.center(10)

    Returns:
      str, The colorized string that can be printed to the console.
    """
    if justify:
        string = justify(string)
    if self._csi and color in self._ANSI_COLOR:
        return '{csi}{color_code}{string}{csi}{reset_code}'.format(csi=self._csi, color_code=self._ANSI_COLOR[color], reset_code=self._ANSI_COLOR_RESET, string=string)
    return string