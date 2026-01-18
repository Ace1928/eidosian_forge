from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..util import AtomicString
import re
import xml.etree.ElementTree as etree
class AbbrExtension(Extension):
    """ Abbreviation Extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Insert `AbbrPreprocessor` before `ReferencePreprocessor`. """
        md.parser.blockprocessors.register(AbbrPreprocessor(md.parser), 'abbr', 16)