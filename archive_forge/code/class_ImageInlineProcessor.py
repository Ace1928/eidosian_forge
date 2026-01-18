from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class ImageInlineProcessor(LinkInlineProcessor):
    """ Return a `img` element from the given match. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """ Return an `img` [`Element`][xml.etree.ElementTree.Element] or `(None, None, None)`. """
        text, index, handled = self.getText(data, m.end(0))
        if not handled:
            return (None, None, None)
        src, title, index, handled = self.getLink(data, index)
        if not handled:
            return (None, None, None)
        el = etree.Element('img')
        el.set('src', src)
        if title is not None:
            el.set('title', title)
        el.set('alt', self.unescape(text))
        return (el, m.start(0), index)