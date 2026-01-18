from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _NewLine(self, overflow=None):
    """Adds the current token list to the line list.

    Args:
      overflow: If not None then this is a (word, available) tuple from Fill()
        where word caused the line width overflow and available is the number of
        characters available in the current line before ' '+word would be
        appended.
    """
    tokens = self._tokens
    self._tokens = []
    if self._truncated or (not tokens and self._compact):
        return
    if self._lines:
        while self._lines[-1] and self._lines[-1][-1][self.TOKEN_TEXT_INDEX].isspace():
            self._lines[-1] = self._lines[-1][:-1]
    if self._height and len(self._lines) + int(bool(tokens)) >= self._height:
        tokens = self._Truncate(tokens, overflow)
    self._lines.append(tokens)