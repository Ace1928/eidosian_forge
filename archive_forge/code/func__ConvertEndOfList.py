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
def _ConvertEndOfList(self, i):
    """Detects and converts an end of list markdown line.

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is an end of list markdown, i otherwise.
    """
    if i or not self._depth:
        return i
    if not self._lists[self._depth].line_break_seen:
        return i
    if self._lists[self._depth].ignore_line > 1:
        self._lists[self._depth].ignore_line -= 1
    if not self._lists[self._depth].ignore_line:
        self._Fill()
        self._renderer.List(self._depth - 1, end=True)
    return i