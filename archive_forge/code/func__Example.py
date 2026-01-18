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
def _Example(self, i):
    """Renders self._line[i:] as an example.

    This is a helper function for _ConvertCodeBlock() and _ConvertExample().

    Args:
      i: The current character index in self._line.
    """
    if self._line[i:]:
        self._Fill()
        if not self._example or self._example > i:
            self._example = i
        self._next_example = self._example
        self._buf = self._line[self._example:]
        self._renderer.Example(self._Attributes())