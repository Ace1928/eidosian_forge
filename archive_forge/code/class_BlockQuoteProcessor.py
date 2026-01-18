from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class BlockQuoteProcessor(BlockProcessor):
    """ Process blockquotes. """
    RE = re.compile('(^|\\n)[ ]{0,3}>[ ]?(.*)')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block)) and (not util.nearing_recursion_limit())

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]
            self.parser.parseBlocks(parent, [before])
            block = '\n'.join([self.clean(line) for line in block[m.start():].split('\n')])
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag == 'blockquote':
            quote = sibling
        else:
            quote = etree.SubElement(parent, 'blockquote')
        self.parser.state.set('blockquote')
        self.parser.parseChunk(quote, block)
        self.parser.state.reset()

    def clean(self, line: str) -> str:
        """ Remove `>` from beginning of a line. """
        m = self.RE.match(line)
        if line.strip() == '>':
            return ''
        elif m:
            return m.group(2)
        else:
            return line