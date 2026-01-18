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
class IndexInSectionTitleTransform(SphinxPostTransform):
    """Move index nodes in section title to outside of the title.

    LaTeX index macro is not compatible with some handling of section titles
    such as uppercasing done on LaTeX side (cf. fncychap handling of ``\\chapter``).
    Moving the index node to after the title node fixes that.

    Before::

        <section>
            <title>
                blah blah <index entries=[...]/>blah
            <paragraph>
                blah blah blah
            ...

    After::

        <section>
            <title>
                blah blah blah
            <index entries=[...]/>
            <paragraph>
                blah blah blah
            ...
    """
    default_priority = 400
    formats = ('latex',)

    def run(self, **kwargs: Any) -> None:
        for node in list(self.document.findall(nodes.title)):
            if isinstance(node.parent, nodes.section):
                for i, index in enumerate(node.findall(addnodes.index)):
                    node.remove(index)
                    node.parent.insert(i + 1, index)