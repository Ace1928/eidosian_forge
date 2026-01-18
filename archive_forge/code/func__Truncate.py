from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import renderer
from prompt_toolkit.token import Token
def _Truncate(self, tokens, overflow):
    """Injects a truncation indicator token and rejects subsequent tokens.

    Args:
      tokens: The last line of tokens at the output height. The part of the
        line within the output width will be visible, modulo the trailing
        truncation marker token added here.
      overflow: If not None then this is a (word, available) tuple from Fill()
        where word caused the line width overflow and available is the number of
        characters available in the current line before ' '+word would be
        appended.

    Returns:
      A possibly altered list of tokens that form the last output line.
    """
    self._truncated = True
    marker_string = '...'
    marker_width = len(marker_string)
    marker_token = (Token.Markdown.Truncated, marker_string)
    if tokens and overflow:
        word, available = overflow
        if marker_width == available:
            pass
        elif marker_width + 1 <= available:
            word = ' ' + self._UnFormat(word)[:available - marker_width - 1]
            tokens.append((self._current_token_type, word))
        else:
            truncated_tokens = []
            available = self._width
            for token in tokens:
                word = token[self.TOKEN_TEXT_INDEX]
                width = self._attr.DisplayWidth(word)
                available -= width
                if available <= marker_width:
                    trim = marker_width - available
                    if trim:
                        word = word[:-trim]
                    truncated_tokens.append((token[self.TOKEN_TYPE_INDEX], word))
                    break
                truncated_tokens.append(token)
            tokens = truncated_tokens
    tokens.append(marker_token)
    return tokens