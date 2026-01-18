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
class ApplySourceWorkaround(SphinxTransform):
    """
    Update source and rawsource attributes
    """
    default_priority = 10

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall():
            if isinstance(node, (nodes.TextElement, nodes.image, nodes.topic)):
                apply_source_workaround(node)