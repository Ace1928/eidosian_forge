from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
def _SkipControlSequence(self, line, index):
    """Skip the control sequence at line[index].

    Args:
      line: The string.
      index: The starting index in string.

    Returns:
      The index in line after the control sequence or len(line) at end of
      string.
    """
    n = self._attr.GetControlSequenceLen(line[index:])
    if not n:
        n = 1
    return index + n