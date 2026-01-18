from __future__ import annotations
import re
from markdown.treeprocessors import Treeprocessor, isString
from markdown.extensions import Extension
from typing import TYPE_CHECKING
class LegacyAttrExtension(Extension):

    def extendMarkdown(self, md):
        """ Add `LegacyAttrs` to Markdown instance. """
        md.treeprocessors.register(LegacyAttrs(md), 'legacyattrs', 15)