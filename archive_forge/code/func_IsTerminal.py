from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def IsTerminal(self):
    """True if the resource has finished the last operation, for good or ill.

    conditions are considered terminal if and only if the ready condition is
    either true or false.

    Returns:
      A bool representing if terminal.
    """
    if not self._ready_condition:
        raise NotImplementedError()
    if not self._fresh:
        return False
    if self._ready_condition not in self._conditions:
        return False
    return self._conditions[self._ready_condition]['status'] is not None