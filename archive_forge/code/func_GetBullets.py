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
def GetBullets(self):
    """Returns the bullet characters list.

    Use the list elements in order for best appearance in nested bullet lists,
    wrapping back to the first element for deep nesting. The list size depends
    on the console implementation.

    Returns:
      A tuple of bullet characters.
    """
    return self._bullets