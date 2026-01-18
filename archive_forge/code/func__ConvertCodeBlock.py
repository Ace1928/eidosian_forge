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
def _ConvertCodeBlock(self, i):
    """Detects and converts a ```...``` code block markdown.

    Args:
      i: The current character index in self._line.

    Returns:
      -1 if the input line is part of a code block markdown, i otherwise.
    """
    if self._line[i:].startswith('```'):
        lang = self._line[i + 3:]
        if not lang:
            if self._code_block_indent >= 0:
                self._code_block_indent = -1
            else:
                self._code_block_indent = i
            self._renderer.SetLang('' if self._code_block_indent >= 0 else None)
            return -1
        if self._code_block_indent < 0 and lang.isalnum():
            self._renderer.SetLang(lang)
            self._code_block_indent = i
            return -1
    if self._code_block_indent < 0:
        return i
    self._Example(self._code_block_indent)
    return -1