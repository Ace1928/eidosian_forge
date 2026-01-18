from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
class AttrListExtension(Extension):
    """ Attribute List extension for Python-Markdown """

    def extendMarkdown(self, md):
        md.treeprocessors.register(AttrListTreeprocessor(md), 'attr_list', 8)
        md.registerExtension(self)