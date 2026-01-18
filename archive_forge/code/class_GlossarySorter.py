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
class GlossarySorter(SphinxTransform):
    """Sort glossaries that have the ``sorted`` flag."""
    default_priority = 500

    def apply(self, **kwargs: Any) -> None:
        for glossary in self.document.findall(addnodes.glossary):
            if glossary['sorted']:
                definition_list = cast(nodes.definition_list, glossary[0])
                definition_list[:] = sorted(definition_list, key=lambda item: unicodedata.normalize('NFD', cast(nodes.term, item)[0].astext().lower()))