from __future__ import annotations
import logging
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from .blockparser import BlockParser
class OListProcessor(BlockProcessor):
    """ Process ordered list blocks. """
    TAG: str = 'ol'
    ' The tag used for the the wrapping element. '
    STARTSWITH: str = '1'
    '\n    The integer (as a string ) with which the list starts. For example, if a list is initialized as\n    `3. Item`, then the `ol` tag will be assigned an HTML attribute of `starts="3"`. Default: `"1"`.\n    '
    LAZY_OL: bool = True
    ' Ignore `STARTSWITH` if `True`. '
    SIBLING_TAGS: list[str] = ['ol', 'ul']
    '\n    Markdown does not require the type of a new list item match the previous list item type.\n    This is the list of types which can be mixed.\n    '

    def __init__(self, parser: BlockParser):
        super().__init__(parser)
        self.RE = re.compile('^[ ]{0,%d}\\d+\\.[ ]+(.*)' % (self.tab_length - 1))
        self.CHILD_RE = re.compile('^[ ]{0,%d}((\\d+\\.)|[*+-])[ ]+(.*)' % (self.tab_length - 1))
        self.INDENT_RE = re.compile('^[ ]{%d,%d}((\\d+\\.)|[*+-])[ ]+.*' % (self.tab_length, self.tab_length * 2 - 1))

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.match(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> None:
        items = self.get_items(blocks.pop(0))
        sibling = self.lastChild(parent)
        if sibling is not None and sibling.tag in self.SIBLING_TAGS:
            lst = sibling
            if lst[-1].text:
                p = etree.Element('p')
                p.text = lst[-1].text
                lst[-1].text = ''
                lst[-1].insert(0, p)
            lch = self.lastChild(lst[-1])
            if lch is not None and lch.tail:
                p = etree.SubElement(lst[-1], 'p')
                p.text = lch.tail.lstrip()
                lch.tail = ''
            li = etree.SubElement(lst, 'li')
            self.parser.state.set('looselist')
            firstitem = items.pop(0)
            self.parser.parseBlocks(li, [firstitem])
            self.parser.state.reset()
        elif parent.tag in ['ol', 'ul']:
            lst = parent
        else:
            lst = etree.SubElement(parent, self.TAG)
            if not self.LAZY_OL and self.STARTSWITH != '1':
                lst.attrib['start'] = self.STARTSWITH
        self.parser.state.set('list')
        for item in items:
            if item.startswith(' ' * self.tab_length):
                self.parser.parseBlocks(lst[-1], [item])
            else:
                li = etree.SubElement(lst, 'li')
                self.parser.parseBlocks(li, [item])
        self.parser.state.reset()

    def get_items(self, block: str) -> list[str]:
        """ Break a block into list items. """
        items = []
        for line in block.split('\n'):
            m = self.CHILD_RE.match(line)
            if m:
                if not items and self.TAG == 'ol':
                    INTEGER_RE = re.compile('(\\d+)')
                    self.STARTSWITH = INTEGER_RE.match(m.group(1)).group()
                items.append(m.group(3))
            elif self.INDENT_RE.match(line):
                if items[-1].startswith(' ' * self.tab_length):
                    items[-1] = '{}\n{}'.format(items[-1], line)
                else:
                    items.append(line)
            else:
                items[-1] = '{}\n{}'.format(items[-1], line)
        return items