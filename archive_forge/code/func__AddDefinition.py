from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _AddDefinition(self, text):
    """Appends a definition list definition item to the current line."""
    text = self._UnFormat(text)
    parts = text.split('=', 1)
    self._AddToken(parts[0], Token.Markdown.Definition)
    if len(parts) > 1:
        self._AddToken('=', Token.Markdown.Normal)
        self._AddToken(parts[1], Token.Markdown.Value)
    self._NewLine()