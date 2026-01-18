from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor, ListIndentProcessor
import xml.etree.ElementTree as etree
import re
class DefListProcessor(BlockProcessor):
    """ Process Definition Lists. """
    RE = re.compile('(^|\\n)[ ]{0,3}:[ ]{1,3}(.*?)(\\n|$)')
    NO_INDENT_RE = re.compile('^[ ]{0,3}[^ :]')

    def test(self, parent: etree.Element, block: str) -> bool:
        return bool(self.RE.search(block))

    def run(self, parent: etree.Element, blocks: list[str]) -> bool | None:
        raw_block = blocks.pop(0)
        m = self.RE.search(raw_block)
        terms = [term.strip() for term in raw_block[:m.start()].split('\n') if term.strip()]
        block = raw_block[m.end():]
        no_indent = self.NO_INDENT_RE.match(block)
        if no_indent:
            d, theRest = (block, None)
        else:
            d, theRest = self.detab(block)
        if d:
            d = '{}\n{}'.format(m.group(2), d)
        else:
            d = m.group(2)
        sibling = self.lastChild(parent)
        if not terms and sibling is None:
            blocks.insert(0, raw_block)
            return False
        if not terms and sibling.tag == 'p':
            state = 'looselist'
            terms = sibling.text.split('\n')
            parent.remove(sibling)
            sibling = self.lastChild(parent)
        else:
            state = 'list'
        if sibling is not None and sibling.tag == 'dl':
            dl = sibling
            if not terms and len(dl) and (dl[-1].tag == 'dd') and len(dl[-1]):
                state = 'looselist'
        else:
            dl = etree.SubElement(parent, 'dl')
        for term in terms:
            dt = etree.SubElement(dl, 'dt')
            dt.text = term
        self.parser.state.set(state)
        dd = etree.SubElement(dl, 'dd')
        self.parser.parseBlocks(dd, [d])
        self.parser.state.reset()
        if theRest:
            blocks.insert(0, theRest)