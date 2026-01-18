from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class ReferenceInlineProcessor(LinkInlineProcessor):
    """ Match to a stored reference and return link element. """
    NEWLINE_CLEANUP_RE = re.compile('\\s+', re.MULTILINE)
    RE_LINK = re.compile('\\s?\\[([^\\]]*)\\]', re.DOTALL | re.UNICODE)

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | None, int | None, int | None]:
        """
        Return [`Element`][xml.etree.ElementTree.Element] returned by `makeTag` method or `(None, None, None)`.

        """
        text, index, handled = self.getText(data, m.end(0))
        if not handled:
            return (None, None, None)
        id, end, handled = self.evalId(data, index, text)
        if not handled:
            return (None, None, None)
        id = self.NEWLINE_CLEANUP_RE.sub(' ', id)
        if id not in self.md.references:
            return (None, m.start(0), end)
        href, title = self.md.references[id]
        return (self.makeTag(href, title, text), m.start(0), end)

    def evalId(self, data: str, index: int, text: str) -> tuple[str | None, int, bool]:
        """
        Evaluate the id portion of `[ref][id]`.

        If `[ref][]` use `[ref]`.
        """
        m = self.RE_LINK.match(data, pos=index)
        if not m:
            return (None, index, False)
        else:
            id = m.group(1).lower()
            end = m.end(0)
            if not id:
                id = text.lower()
        return (id, end, True)

    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        """ Return an `a` [`Element`][xml.etree.ElementTree.Element]. """
        el = etree.Element('a')
        el.set('href', href)
        if title:
            el.set('title', title)
        el.text = text
        return el