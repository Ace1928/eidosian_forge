from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _GetNumMatchingLines(self, new_lines):
    matching_lines = 0
    for i in range(min(len(new_lines), self._num_lines)):
        if new_lines[i] != self._lines[i]:
            break
        matching_lines += 1
    return matching_lines