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
class MathReferenceTransform(SphinxPostTransform):
    """Replace pending_xref nodes for math by math_reference.

    To handle math reference easily on LaTeX writer, this converts pending_xref
    nodes to math_reference.
    """
    default_priority = 5
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        equations = self.env.get_domain('math').data['objects']
        for node in self.document.findall(addnodes.pending_xref):
            if node['refdomain'] == 'math' and node['reftype'] in ('eq', 'numref'):
                docname, _ = equations.get(node['reftarget'], (None, None))
                if docname:
                    refnode = math_reference('', docname=docname, target=node['reftarget'])
                    node.replace_self(refnode)