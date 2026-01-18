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
class CitationReferenceTransform(SphinxPostTransform):
    """Replace pending_xref nodes for citation by citation_reference.

    To handle citation reference easily on LaTeX writer, this converts
    pending_xref nodes to citation_reference.
    """
    default_priority = 5
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        domain = cast(CitationDomain, self.env.get_domain('citation'))
        matcher = NodeMatcher(addnodes.pending_xref, refdomain='citation', reftype='ref')
        for node in self.document.findall(matcher):
            docname, labelid, _ = domain.citations.get(node['reftarget'], ('', '', 0))
            if docname:
                citation_ref = nodes.citation_reference('', '', *node.children, docname=docname, refname=labelid)
                node.replace_self(citation_ref)