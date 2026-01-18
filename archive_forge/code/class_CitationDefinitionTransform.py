from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast
from docutils import nodes
from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.nodes import copy_source_info, make_refnode
class CitationDefinitionTransform(SphinxTransform):
    """Mark citation definition labels as not smartquoted."""
    default_priority = 619

    def apply(self, **kwargs: Any) -> None:
        domain = cast(CitationDomain, self.env.get_domain('citation'))
        for node in self.document.findall(nodes.citation):
            node['docname'] = self.env.docname
            domain.note_citation(node)
            label = cast(nodes.label, node[0])
            label['support_smartquotes'] = False