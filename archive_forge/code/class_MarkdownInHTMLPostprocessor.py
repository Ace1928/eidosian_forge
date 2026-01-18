from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
class MarkdownInHTMLPostprocessor(RawHtmlPostprocessor):

    def stash_to_string(self, text: str | etree.Element) -> str:
        """ Override default to handle any `etree` elements still in the stash. """
        if isinstance(text, etree.Element):
            return self.md.serializer(text)
        else:
            return str(text)