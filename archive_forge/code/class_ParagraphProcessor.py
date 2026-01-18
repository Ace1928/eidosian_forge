from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class ParagraphProcessor(BlockProcessor):
    """ Process Paragraph blocks. """

    def test(self, parent: etree.Element, block: str) -> bool:
        return True

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        if block.strip():
            if self.parser.state.isstate('list'):
                sibling = self.lastChild(parent)
                if sibling is not None:
                    if sibling.tail:
                        sibling.tail = '{}\n{}'.format(sibling.tail, block)
                    else:
                        sibling.tail = '\n%s' % block
                elif parent.text:
                    parent.text = '{}\n{}'.format(parent.text, block)
                else:
                    parent.text = block.lstrip()
            else:
                p = etree.SubElement(parent, 'p')
                p.text = block.lstrip()