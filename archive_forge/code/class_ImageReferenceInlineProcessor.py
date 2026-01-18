from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class ImageReferenceInlineProcessor(ReferenceInlineProcessor):
    """ Match to a stored reference and return `img` element. """

    def makeTag(self, href: str, title: str, text: str) -> etree.Element:
        """ Return an `img` [`Element`][xml.etree.ElementTree.Element]. """
        el = etree.Element('img')
        el.set('src', href)
        if title:
            el.set('title', title)
        el.set('alt', self.unescape(text))
        return el