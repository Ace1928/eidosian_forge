from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class HtmlInlineProcessor(InlineProcessor):
    """ Store raw inline html and return a placeholder. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        """ Store the text of `group(1)` of a pattern and return a placeholder string. """
        rawhtml = self.backslash_unescape(self.unescape(m.group(1)))
        place_holder = self.md.htmlStash.store(rawhtml)
        return (place_holder, m.start(0), m.end(0))

    def unescape(self, text: str) -> str:
        """ Return unescaped text given text with an inline placeholder. """
        try:
            stash = self.md.treeprocessors['inline'].stashed_nodes
        except KeyError:
            return text

        def get_stash(m: re.Match[str]) -> str:
            id = m.group(1)
            value = stash.get(id)
            if value is not None:
                try:
                    return self.md.serializer(value)
                except Exception:
                    return '\\%s' % value
        return util.INLINE_PLACEHOLDER_RE.sub(get_stash, text)

    def backslash_unescape(self, text: str) -> str:
        """ Return text with backslash escapes undone (backslashes are restored). """
        try:
            RE = self.md.treeprocessors['unescape'].RE
        except KeyError:
            return text

        def _unescape(m: re.Match[str]) -> str:
            return chr(int(m.group(1)))
        return RE.sub(_unescape, text)