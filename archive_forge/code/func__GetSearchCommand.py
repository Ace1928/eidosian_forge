from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import sys
from googlecloudsdk.core.console import console_attr
def _GetSearchCommand(self, c):
    """Consumes a search command and returns the equivalent pager command.

    The search pattern is an RE that is pre-compiled and cached for subsequent
    /<newline>, ?<newline>, n, or N commands.

    Args:
      c: The search command char.

    Returns:
      The pager command char.
    """
    self._Write(c)
    buf = ''
    while True:
        p = self._attr.GetRawKey()
        if p in (None, '\n', '\r') or len(p) != 1:
            break
        self._Write(p)
        buf += p
    self._Write('\r' + ' ' * len(buf) + '\r')
    if buf:
        try:
            self._search_pattern = re.compile(buf)
        except re.error:
            self._search_pattern = None
            return ''
    self._search_direction = 'n' if c == '/' else 'N'
    return 'n'