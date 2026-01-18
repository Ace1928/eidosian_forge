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
def ConvertOutputToUnicode(self, buf):
    """Converts a console output string buf to unicode.

    Mainly used for testing. Allows test comparisons in unicode while ensuring
    that unicode => encoding => unicode works.

    Args:
      buf: The console output string to convert.

    Returns:
      The console output string buf converted to unicode.
    """
    if isinstance(buf, six.text_type):
        buf = buf.encode(self._encoding)
    return six.text_type(buf, self._encoding, 'replace')