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
def _ConvertBulletList(self, i):
    """Detects and converts a bullet list item markdown line.

    The list item indicator may be '-' or '*'. nesting by multiple indicators:

        - level-1
        -- level-2
        - level-1

    or nesting by indicator indentation:

        * level-1
          * level-2
        * level-1

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is a bullet list item markdown, i otherwise.
    """
    if self._example or self._line[i] not in '-*':
        return i
    bullet = self._line[i]
    level = i / 2
    start_index = i
    while i < len(self._line) and self._line[i] == bullet:
        i += 1
        level += 1
    if i >= len(self._line) or self._line[i] != ' ':
        return start_index
    if self._lists[self._depth].bullet and self._lists[self._depth].level >= level:
        while self._lists[self._depth].level > level:
            self._depth -= 1
    else:
        self._depth += 1
        if self._depth >= len(self._lists):
            self._lists.append(_ListElementState())
    self._lists[self._depth].bullet = True
    self._lists[self._depth].ignore_line = 0
    self._lists[self._depth].line_break_seen = False
    self._lists[self._depth].level = level
    self._Fill()
    self._renderer.List(self._depth)
    while i < len(self._line) and self._line[i] == ' ':
        i += 1
    self._buf += self._line[i:]
    return -1