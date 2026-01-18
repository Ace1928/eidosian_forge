import re
import unicodedata
import warnings
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple, cast
import docutils
from docutils import nodes
from docutils.nodes import Element, Node, Text
from docutils.transforms import Transform, Transformer
from docutils.transforms.parts import ContentsFilter
from docutils.transforms.universal import SmartQuotes
from docutils.utils import normalize_language_tag
from docutils.utils.smartquotes import smartchars
from sphinx import addnodes
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.docutils import new_document
from sphinx.util.i18n import format_date
from sphinx.util.nodes import NodeMatcher, apply_source_workaround, is_smartquotable
class ManpageLink(SphinxTransform):
    """Find manpage section numbers and names"""
    default_priority = 999

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.manpage):
            manpage = ' '.join([str(x) for x in node.children if isinstance(x, nodes.Text)])
            pattern = '^(?P<path>(?P<page>.+)[\\(\\.](?P<section>[1-9]\\w*)?\\)?)$'
            info = {'path': manpage, 'page': manpage, 'section': ''}
            r = re.match(pattern, manpage)
            if r:
                info = r.groupdict()
            node.attributes.update(info)