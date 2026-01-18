from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class SimpleTextInlineProcessor(InlineProcessor):
    """ Return a simple text of `group(1)` of a Pattern. """

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[str, int, int]:
        """ Return string content of `group(1)` of a matching pattern. """
        return (m.group(1), m.start(0), m.end(0))