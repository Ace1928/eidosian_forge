from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _AddToken(self, text, token_type=None):
    """Appends a (token_type, text) tuple to the current line."""
    if text and (not text.isspace()):
        self._ignore_paragraph = False
    if not token_type:
        token_type = self._current_token_type
    if self._csi not in text:
        self._MergeOrAddToken(text, token_type)
    else:
        i = 0
        while True:
            j = text.find(self._csi, i)
            if j < 0:
                self._MergeOrAddToken(text[i:], token_type)
                break
            self._MergeOrAddToken(text[i:j], token_type)
            token_type = self.EMBELLISHMENTS[text[j + 1]]
            self._current_token_type = token_type
            i = j + 2