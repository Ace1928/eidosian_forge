from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
class HtmlBlockPreprocessor(Preprocessor):
    """Remove html blocks from the text and store them for later retrieval."""

    def run(self, lines: list[str]) -> list[str]:
        source = '\n'.join(lines)
        parser = HTMLExtractorExtra(self.md)
        parser.feed(source)
        parser.close()
        return ''.join(parser.cleandoc).split('\n')