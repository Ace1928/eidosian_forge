from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class EmptyBlockProcessor(BlockProcessor):
    """ Process blocks that are empty or start with an empty line. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return not block or block.startswith('\n')

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        filler = '\n\n'
        if block:
            filler = '\n'
            theRest = block[1:]
            if theRest:
                blocks.insert(0, theRest)
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag == 'pre' and len(sibling) and (sibling[0].tag == 'code'):
            sibling[0].text = util.AtomicString('{}{}'.format(sibling[0].text, filler))