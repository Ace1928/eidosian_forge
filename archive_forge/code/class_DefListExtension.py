from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor, ListIndentProcessor
import xml.etree.ElementTree as etree
import re
class DefListExtension(Extension):
    """ Add definition lists to Markdown. """

    def extendMarkdown(self, md):
        """ Add an instance of `DefListProcessor` to `BlockParser`. """
        md.parser.blockprocessors.register(DefListIndentProcessor(md.parser), 'defindent', 85)
        md.parser.blockprocessors.register(DefListProcessor(md.parser), 'deflist', 25)