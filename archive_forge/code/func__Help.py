from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.core.console import console_attr
def _Help(self):
    """Print command help and wait for any character to continue."""
    clear = self._height - (len(self.HELP_TEXT) - len(self.HELP_TEXT.replace('\n', '')))
    if clear > 0:
        self._Write('\n' * clear)
    self._Write(self.HELP_TEXT)
    self._attr.GetRawKey()
    self._Write('\n')