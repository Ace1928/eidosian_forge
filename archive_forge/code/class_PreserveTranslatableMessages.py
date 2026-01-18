from os import path
from re import DOTALL, match
from textwrap import indent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, TypeVar
from docutils import nodes
from docutils.io import StringInput
from docutils.nodes import Element
from docutils.utils import relative_path
from sphinx import addnodes
from sphinx.config import Config
from sphinx.domains.std import make_glossary_term, split_term_classifiers
from sphinx.locale import __
from sphinx.locale import init as init_locale
from sphinx.transforms import SphinxTransform
from sphinx.util import get_filetype, logging, split_index_msg
from sphinx.util.i18n import docname_to_domain
from sphinx.util.nodes import (IMAGE_TYPE_NODES, LITERAL_TYPE_NODES, NodeMatcher,
class PreserveTranslatableMessages(SphinxTransform):
    """
    Preserve original translatable messages before translation
    """
    default_priority = 10

    def apply(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.translatable):
            node.preserve_original_messages()