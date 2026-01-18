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
def GetFontCode(self, bold=False, italic=False):
    """Returns a font code string for 0 or more embellishments.

    GetFontCode() with no args returns the default font code string.

    Args:
      bold: True for bold embellishment.
      italic: True for italic embellishment.

    Returns:
      The font code string for the requested embellishments. Write this string
        to the console output to control the font settings.
    """
    if not self._csi:
        return ''
    codes = []
    if bold:
        codes.append(self._font_bold)
    if italic:
        codes.append(self._font_italic)
    return '{csi}{codes}m'.format(csi=self._csi, codes=';'.join(codes))