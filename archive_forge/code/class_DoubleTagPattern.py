from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class DoubleTagPattern(SimpleTagPattern):
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """

    def handleMatch(self, m: re.Match[str]) -> etree.Element:
        """
        Return [`Element`][xml.etree.ElementTree.Element] in following format:
        `<tag1><tag2>group(3)</tag2>group(4)</tag2>` where `group(4)` is optional.

        """
        tag1, tag2 = self.tag.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(3)
        if len(m.groups()) == 5:
            el2.tail = m.group(4)
        return el1