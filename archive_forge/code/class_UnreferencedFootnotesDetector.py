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
class UnreferencedFootnotesDetector(SphinxTransform):
    """
    Detect unreferenced footnotes and emit warnings
    """
    default_priority = 200

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.footnotes:
            if node['names'] == []:
                pass
            elif node['names'][0] not in self.document.footnote_refs:
                logger.warning(__('Footnote [%s] is not referenced.'), node['names'][0], type='ref', subtype='footnote', location=node)
        for node in self.document.autofootnotes:
            if not any((ref['auto'] == node['auto'] for ref in self.document.autofootnote_refs)):
                logger.warning(__('Footnote [#] is not referenced.'), type='ref', subtype='footnote', location=node)