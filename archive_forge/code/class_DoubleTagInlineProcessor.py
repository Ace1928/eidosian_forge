from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class DoubleTagInlineProcessor(SimpleTagInlineProcessor):
    """Return a ElementTree element nested in tag2 nested in tag1.

    Useful for strong emphasis etc.

    """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """
        Return [`Element`][xml.etree.ElementTree.Element] in following format:
        `<tag1><tag2>group(2)</tag2>group(3)</tag2>` where `group(3)` is optional.

        """
        tag1, tag2 = self.tag.split(',')
        el1 = etree.Element(tag1)
        el2 = etree.SubElement(el1, tag2)
        el2.text = m.group(2)
        if len(m.groups()) == 3:
            el2.tail = m.group(3)
        return (el1, m.start(0), m.end(0))