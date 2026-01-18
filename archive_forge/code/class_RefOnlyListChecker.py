from typing import Any, Dict, List, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform
class RefOnlyListChecker(nodes.GenericNodeVisitor):
    """Raise `nodes.NodeFound` if non-simple list item is encountered.

    Here 'simple' means a list item containing only a paragraph with a
    single reference in it.
    """

    def default_visit(self, node: Node) -> None:
        raise nodes.NodeFound

    def visit_bullet_list(self, node: nodes.bullet_list) -> None:
        pass

    def visit_list_item(self, node: nodes.list_item) -> None:
        children: List[Node] = []
        for child in node.children:
            if not isinstance(child, nodes.Invisible):
                children.append(child)
        if len(children) != 1:
            raise nodes.NodeFound
        if not isinstance(children[0], nodes.paragraph):
            raise nodes.NodeFound
        para = children[0]
        if len(para) != 1:
            raise nodes.NodeFound
        if not isinstance(para[0], addnodes.pending_xref):
            raise nodes.NodeFound
        raise nodes.SkipChildren

    def invisible_visit(self, node: Node) -> None:
        """Invisible nodes should be ignored."""
        pass