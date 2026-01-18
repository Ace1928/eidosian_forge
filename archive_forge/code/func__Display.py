from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _Display(self, prefix, matches):
    """Displays the possible completions and redraws the prompt and response.

    Args:
      prefix: str, The current response.
      matches: [str], The list of strings that start with prefix.
    """
    row_data = _TransposeListToRows(matches, width=self._width, height=self._height, pad=self._pad, bold=self._attr.GetFontCode(bold=True), normal=self._attr.GetFontCode())
    if self._prompt:
        row_data.append(self._prompt)
    row_data.append(prefix)
    self._out.write(''.join(row_data))