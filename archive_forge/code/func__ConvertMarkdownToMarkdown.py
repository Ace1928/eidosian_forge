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
def _ConvertMarkdownToMarkdown(self):
    """Generates markdown with additonal NOTES if requested."""
    if not self._edit:
        self._renderer.Write(self._fin.read())
        return
    while True:
        line = self._ReadLine()
        if not line:
            break
        self._renderer.Write(line)
        if self._notes and line == '## NOTES\n':
            self._renderer.Write('\n' + self._notes + '\n')
            self._notes = ''
    if self._notes:
        self._renderer.Write('\n\n## NOTES\n\n' + self._notes + '\n')