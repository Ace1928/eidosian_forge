from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class HashHeaderProcessor(BlockProcessor):
    """ Process Hash Headers. """
    RE = re.compile('(?:^|\\n)(?P<level>#{1,6})(?P<header>(?:\\\\.|[^\\\\])*?)#*(?:\\n|$)')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        m = self.RE.search(block)
        if m:
            before = block[:m.start()]
            after = block[m.end():]
            if before:
                self.parser.parseBlocks(parent, [before])
            h = etree.SubElement(parent, 'h%d' % len(m.group('level')))
            h.text = m.group('header').strip()
            if after:
                if self.parser.state.isstate('looselist'):
                    after = self.looseDetab(after)
                blocks.insert(0, after)
        else:
            logger.warn("We've got a problem header: %r" % block)