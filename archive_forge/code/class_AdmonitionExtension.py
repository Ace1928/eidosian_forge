from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING
class AdmonitionExtension(Extension):
    """ Admonition extension for Python-Markdown. """

    def extendMarkdown(self, md):
        """ Add Admonition to Markdown instance. """
        md.registerExtension(self)
        md.parser.blockprocessors.register(AdmonitionProcessor(md.parser), 'admonition', 105)