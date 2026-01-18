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
def _GetConsoleEncoding(self):
    """Gets the encoding as declared by the stdout stream.

    Returns:
      str, The encoding name or None if it could not be determined.
    """
    console_encoding = getattr(sys.stdout, 'encoding', None)
    if not console_encoding:
        return None
    console_encoding = console_encoding.lower()
    if 'utf-8' in console_encoding:
        locale_encoding = locale.getpreferredencoding()
        if locale_encoding and 'cp1252' in locale_encoding:
            return None
        return 'utf-8'
    elif 'cp437' in console_encoding:
        return 'cp437'
    elif 'cp1252' in console_encoding:
        return None
    return None