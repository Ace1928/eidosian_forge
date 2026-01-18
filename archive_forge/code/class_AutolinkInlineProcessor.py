from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class AutolinkInlineProcessor(InlineProcessor):
    """ Return a link Element given an auto-link (`<http://example/com>`). """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element, int, int]:
        """ Return an `a` [`Element`][xml.etree.ElementTree.Element] of `group(1)`. """
        el = etree.Element('a')
        el.set('href', self.unescape(m.group(1)))
        el.text = util.AtomicString(m.group(1))
        return (el, m.start(0), m.end(0))