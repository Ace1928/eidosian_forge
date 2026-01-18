from __future__ import annotations
from typing import TYPE_CHECKING, Any
from . import util
from .htmlparser import HTMLExtractor
import re
class NormalizeWhitespace(Preprocessor):
    """ Normalize whitespace for consistent parsing. """

    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        source = source.replace(util.STX, '').replace(util.ETX, '')
        source = source.replace('\r\n', '\n').replace('\r', '\n') + '\n\n'
        source = source.expandtabs(self.md.tab_length)
        source = re.sub('(?<=\\n) +\\n', '\n', source)
        return source.split('\n')