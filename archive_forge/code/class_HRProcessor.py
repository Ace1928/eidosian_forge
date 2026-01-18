from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class HRProcessor(BlockProcessor):
    """ Process Horizontal Rules. """
    RE = '^[ ]{0,3}(?=(?P<atomicgroup>(-+[ ]{0,2}){3,}|(_+[ ]{0,2}){3,}|(\\*+[ ]{0,2}){3,}))(?P=atomicgroup)[ ]*$'
    SEARCH_RE = re.compile(RE, re.MULTILINE)

    def test(self, parent: etree.Element, block: str) -> bool:
        m = self.SEARCH_RE.search(block)
        if m:
            self.match = m
            return True
        return False

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        block = blocks.pop(0)
        match = self.match
        prelines = block[:match.start()].rstrip('\n')
        if prelines:
            self.parser.parseBlocks(parent, [prelines])
        etree.SubElement(parent, 'hr')
        postlines = block[match.end():].lstrip('\n')
        if postlines:
            blocks.insert(0, postlines)