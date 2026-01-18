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
def _Attributes(self, buf=None):
    """Converts inline markdown attributes in self._buf.

    Args:
      buf: Convert markdown from this string instead of self._buf.

    Returns:
      A string with markdown attributes converted to render properly.
    """
    emphasis = '' if self._code_block_indent >= 0 or self._example else '*_`'
    ret = ''
    if buf is None:
        buf = self._buf
        self._buf = ''
    if buf:
        buf = self._renderer.Escape(buf)
        i = 0
        is_literal = False
        while i < len(buf):
            c = buf[i]
            if c == ':':
                index_after_anchor, back, target, text = self._AnchorStyle1(buf, i)
                if index_after_anchor and _IsValidTarget(target) and (not _IsFlagValueLink(buf, i)):
                    ret = ret[:-back]
                    i = index_after_anchor - 1
                    c = self._renderer.Link(target, text)
            elif c == '[':
                index_after_anchor, target, text = self._AnchorStyle2(buf, i)
                if index_after_anchor and _IsValidTarget(target):
                    i = index_after_anchor - 1
                    c = self._renderer.Link(target, text)
            elif c in emphasis:
                l = buf[i - 1] if i else ' '
                r = buf[i + 1] if i < len(buf) - 1 else ' '
                if l != '`' and c == '`' and (r == '`'):
                    x = buf[i + 2] if i < len(buf) - 2 else ' '
                    if x == '`':
                        index_at_code_block_quote = buf.find('```', i + 2)
                        if index_at_code_block_quote > 0:
                            ret += self._renderer.Font(renderer.CODE)
                            ret += buf[i + 3:index_at_code_block_quote]
                            ret += self._renderer.Font(renderer.CODE)
                            i = index_at_code_block_quote + 3
                            continue
                    else:
                        index_at_air_quote = buf.find("''", i)
                        if index_at_air_quote > 0:
                            index_at_air_quote += 2
                            ret += buf[i:index_at_air_quote]
                            i = index_at_air_quote
                            continue
                if r == c:
                    c += c
                    i += 1
                elif c == '*' and l in ' /' and (r in ' ./') or (c != '`' and l in ' /' and (r in ' .')):
                    pass
                elif l.isalnum() and r.isalnum():
                    pass
                elif is_literal and c == '*':
                    pass
                else:
                    if c == '`':
                        is_literal = not is_literal
                    c = self._renderer.Font(self._EMPHASIS[c])
            ret += c
            i += 1
    return self._renderer.Entities(ret)