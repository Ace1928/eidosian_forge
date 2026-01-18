from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def IsReady(self):
    """Return True if the resource has succeeded its current operation."""
    if not self.IsTerminal():
        return False
    return self._conditions[self._ready_condition]['status']