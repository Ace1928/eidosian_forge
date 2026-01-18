from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor, ListIndentProcessor
import xml.etree.ElementTree as etree
import re
class DefListIndentProcessor(ListIndentProcessor):
    """ Process indented children of definition list items. """
    ITEM_TYPES = ['dd', 'li']
    ' Include `dd` in list item types. '
    LIST_TYPES = ['dl', 'ol', 'ul']
    ' Include `dl` is list types. '

    def create_item(self, parent: etree.Element, block: str) -> None:
        """ Create a new `dd` or `li` (depending on parent) and parse the block with it as the parent. """
        dd = etree.SubElement(parent, 'dd')
        self.parser.parseBlocks(dd, [block])