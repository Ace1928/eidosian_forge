from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.domains.python import _pseudo_parse_arglist
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import make_id, make_refnode, nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class JSConstructor(JSCallable):
    """Like a callable but with a different prefix."""
    allow_nesting = True

    def get_display_prefix(self) -> List[Node]:
        return [addnodes.desc_sig_keyword('class', 'class'), addnodes.desc_sig_space()]