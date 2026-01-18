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
class BibliographyTransform(SphinxPostTransform):
    """Gather bibliography entries to tail of document.

    Before::

        <document>
            <paragraph>
                blah blah blah
            <citation>
                ...
            <paragraph>
                blah blah blah
            <citation>
                ...
            ...

    After::

        <document>
            <paragraph>
                blah blah blah
            <paragraph>
                blah blah blah
            ...
            <thebibliography>
                <citation>
                    ...
                <citation>
                    ...
    """
    default_priority = 750
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        citations = thebibliography()
        for node in list(self.document.findall(nodes.citation)):
            node.parent.remove(node)
            citations += node
        if len(citations) > 0:
            self.document += citations