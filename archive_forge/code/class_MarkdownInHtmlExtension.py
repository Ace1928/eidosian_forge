from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..preprocessors import Preprocessor
from ..postprocessors import RawHtmlPostprocessor
from .. import util
from ..htmlparser import HTMLExtractor, blank_line_re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Literal, Mapping
class MarkdownInHtmlExtension(Extension):
    """Add Markdown parsing in HTML to Markdown class."""

    def extendMarkdown(self, md):
        """ Register extension instances. """
        md.preprocessors.register(HtmlBlockPreprocessor(md), 'html_block', 20)
        md.parser.blockprocessors.register(MarkdownInHtmlProcessor(md.parser), 'markdown_block', 105)
        md.postprocessors.register(MarkdownInHTMLPostprocessor(md), 'raw_html', 30)