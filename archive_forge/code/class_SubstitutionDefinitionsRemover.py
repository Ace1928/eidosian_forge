from typing import Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.transforms.references import Substitutions
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.latex.nodes import (captioned_literal_block, footnotemark, footnotetext,
from sphinx.domains.citation import CitationDomain
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import NodeMatcher
class SubstitutionDefinitionsRemover(SphinxPostTransform):
    """Remove ``substitution_definition`` nodes from doctrees."""
    default_priority = Substitutions.default_priority + 1
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        for node in list(self.document.findall(nodes.substitution_definition)):
            node.parent.remove(node)