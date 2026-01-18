from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class CodeBlockProcessor(BlockProcessor):
    """ Process code blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return block.startswith(' ' * self.tab_length)

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        sibling = self.lastChild(parent)
        block = blocks.pop(0)
        theRest = ''
        if sibling is not None and sibling.tag == 'pre' and len(sibling) and (sibling[0].tag == 'code'):
            code = sibling[0]
            block, theRest = self.detab(block)
            code.text = util.AtomicString('{}\n{}\n'.format(code.text, util.code_escape(block.rstrip())))
        else:
            pre = etree.SubElement(parent, 'pre')
            code = etree.SubElement(pre, 'code')
            block, theRest = self.detab(block)
            code.text = util.AtomicString('%s\n' % util.code_escape(block.rstrip()))
        if theRest:
            blocks.insert(0, theRest)