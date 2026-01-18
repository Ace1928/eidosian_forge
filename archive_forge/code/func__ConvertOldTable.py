from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
import sys
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.document_renderers import devsite_renderer
from googlecloudsdk.core.document_renderers import html_renderer
from googlecloudsdk.core.document_renderers import linter_renderer
from googlecloudsdk.core.document_renderers import man_renderer
from googlecloudsdk.core.document_renderers import markdown_renderer
from googlecloudsdk.core.document_renderers import renderer
from googlecloudsdk.core.document_renderers import text_renderer
def _ConvertOldTable(self, i):
    """Detects and converts a sequence of markdown table lines.

    This method will consume multiple input lines if the current line is a
    table heading. The table markdown sequence is:

       [...format="csv"...]
       |====*
       col-1-data-item,col-2-data-item...
         ...
       <blank line ends table>

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input lines are table markdown, i otherwise.
    """
    if self._line[0] != '[' or self._line[-1] != ']' or 'format="csv"' not in self._line:
        return i
    line = self._ReadLine()
    if not line:
        return i
    if not line.startswith('|===='):
        self._PushBackLine(line)
        return i
    rows = []
    while True:
        self._buf = self._ReadLine()
        if not self._buf:
            break
        self._buf = self._buf.rstrip()
        if self._buf.startswith('|===='):
            break
        rows.append(self._Attributes().split(','))
    self._buf = ''
    table = renderer.TableAttributes()
    if len(rows) > 1:
        for label in rows[0]:
            table.AddColumn(label=label)
        rows = rows[1:]
    if table.columns and rows:
        self._renderer.Table(table, rows)
    return -1