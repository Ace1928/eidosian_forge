from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def IsFailed(self):
    """"Return True if the resource has failed its current operation."""
    return self.IsTerminal() and (not self.IsReady())