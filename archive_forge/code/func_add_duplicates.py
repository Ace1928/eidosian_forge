from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
def add_duplicates(self, li: etree.Element, duplicates: int) -> None:
    """ Adjust current `li` and add the duplicates: `fnref2`, `fnref3`, etc. """
    for link in li.iter('a'):
        if link.attrib.get('class', '') == 'footnote-backref':
            ref, rest = link.attrib['href'].split(self.footnotes.get_separator(), 1)
            links = []
            for index in range(2, duplicates + 1):
                sib_link = copy.deepcopy(link)
                sib_link.attrib['href'] = '%s%d%s%s' % (ref, index, self.footnotes.get_separator(), rest)
                links.append(sib_link)
                self.offset += 1
            el = list(li)[-1]
            for link in links:
                el.append(link)
            break