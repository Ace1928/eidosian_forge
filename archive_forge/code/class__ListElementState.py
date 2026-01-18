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
class _ListElementState(object):
    """List element state.

  Attributes:
    bullet: True if the current element is a bullet.
    ignore_line: The number of blank line requests to ignore.
    level: List element nesting level counting from 0.
    line_break_seen: True if line break has been seen for bulleted lists.
  """

    def __init__(self):
        self.bullet = False
        self.ignore_line = 0
        self.level = 0
        self.line_break_seen = False