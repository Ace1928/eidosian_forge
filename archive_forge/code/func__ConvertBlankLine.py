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
def _ConvertBlankLine(self, i):
    """Detects and converts a blank markdown line (length 0).

    Resets the indentation to the default and emits a blank line. Multiple
    blank lines are suppressed in the output.

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is a blank markdown, i otherwise.
    """
    if self._line:
        return i
    self._Fill()
    if self._lists[self._depth].bullet:
        self._renderer.List(self._depth - 1, end=True)
        if self._depth:
            self._depth -= 1
        else:
            self._lists[self._depth].bullet = False
    if self._lists[self._depth].ignore_line:
        self._lists[self._depth].ignore_line -= 1
    if not self._lists[self._depth].line_break_seen:
        self._lists[self._depth].line_break_seen = True
    if not self._depth or not self._lists[self._depth].ignore_line or self._lists[self._depth].line_break_seen:
        self._renderer.Line()
    return -1