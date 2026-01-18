from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
class PrettifyTreeprocessor(Treeprocessor):
    """ Add line breaks to the html document. """

    def _prettifyETree(self, elem: etree.Element) -> None:
        """ Recursively add line breaks to `ElementTree` children. """
        i = '\n'
        if self.md.is_block_level(elem.tag) and elem.tag not in ['code', 'pre']:
            if (not elem.text or not elem.text.strip()) and len(elem) and self.md.is_block_level(elem[0].tag):
                elem.text = i
            for e in elem:
                if self.md.is_block_level(e.tag):
                    self._prettifyETree(e)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

    def run(self, root: etree.Element) -> None:
        """ Add line breaks to `Element` object and its children. """
        self._prettifyETree(root)
        brs = root.iter('br')
        for br in brs:
            if not br.tail or not br.tail.strip():
                br.tail = '\n'
            else:
                br.tail = '\n%s' % br.tail
        pres = root.iter('pre')
        for pre in pres:
            if len(pre) and pre[0].tag == 'code':
                code = pre[0]
                if not len(code) and code.text is not None:
                    code.text = util.AtomicString(code.text.rstrip() + '\n')