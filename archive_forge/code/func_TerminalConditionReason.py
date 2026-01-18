from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def TerminalConditionReason(self):
    """Returns the reason of the terminal condition."""
    if self._ready_condition and self._ready_condition in self and self[self._ready_condition]['reason']:
        return self[self._ready_condition]['reason']
    return None