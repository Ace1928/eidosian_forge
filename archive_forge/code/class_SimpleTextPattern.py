from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class SimpleTextPattern(Pattern):
    """ Return a simple text of `group(2)` of a Pattern. """

    def handleMatch(self, m: re.Match[str]) -> str:
        """ Return string content of `group(2)` of a matching pattern. """
        return m.group(2)