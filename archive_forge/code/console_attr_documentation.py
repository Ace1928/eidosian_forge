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
Renders the string as self._color on the console.

    Args:
      stream: The stream to render the string to. The stream given here *must*
        have the same encoding as sys.stdout for this to work properly.
      justify: The justification function, self._justify if None.
    