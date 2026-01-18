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
class LaTeXFootnoteVisitor(nodes.NodeVisitor):

    def __init__(self, document: nodes.document, footnotes: List[nodes.footnote]) -> None:
        self.appeared: Dict[Tuple[str, str], nodes.footnote] = {}
        self.footnotes: List[nodes.footnote] = footnotes
        self.pendings: List[nodes.footnote] = []
        self.table_footnotes: List[nodes.footnote] = []
        self.restricted: Optional[Element] = None
        super().__init__(document)

    def unknown_visit(self, node: Node) -> None:
        pass

    def unknown_departure(self, node: Node) -> None:
        pass

    def restrict(self, node: Element) -> None:
        if self.restricted is None:
            self.restricted = node

    def unrestrict(self, node: Element) -> None:
        if self.restricted == node:
            self.restricted = None
            pos = node.parent.index(node)
            for i, footnote in enumerate(self.pendings):
                fntext = footnotetext('', *footnote.children, ids=footnote['ids'])
                node.parent.insert(pos + i + 1, fntext)
            self.pendings = []

    def visit_figure(self, node: nodes.figure) -> None:
        self.restrict(node)

    def depart_figure(self, node: nodes.figure) -> None:
        self.unrestrict(node)

    def visit_term(self, node: nodes.term) -> None:
        self.restrict(node)

    def depart_term(self, node: nodes.term) -> None:
        self.unrestrict(node)

    def visit_caption(self, node: nodes.caption) -> None:
        self.restrict(node)

    def depart_caption(self, node: nodes.caption) -> None:
        self.unrestrict(node)

    def visit_title(self, node: nodes.title) -> None:
        if isinstance(node.parent, (nodes.section, nodes.table)):
            self.restrict(node)

    def depart_title(self, node: nodes.title) -> None:
        if isinstance(node.parent, nodes.section):
            self.unrestrict(node)
        elif isinstance(node.parent, nodes.table):
            self.table_footnotes += self.pendings
            self.pendings = []
            self.unrestrict(node)

    def visit_thead(self, node: nodes.thead) -> None:
        self.restrict(node)

    def depart_thead(self, node: nodes.thead) -> None:
        self.table_footnotes += self.pendings
        self.pendings = []
        self.unrestrict(node)

    def depart_table(self, node: nodes.table) -> None:
        tbody = next(node.findall(nodes.tbody))
        for footnote in reversed(self.table_footnotes):
            fntext = footnotetext('', *footnote.children, ids=footnote['ids'])
            tbody.insert(0, fntext)
        self.table_footnotes = []

    def visit_footnote(self, node: nodes.footnote) -> None:
        self.restrict(node)

    def depart_footnote(self, node: nodes.footnote) -> None:
        self.unrestrict(node)

    def visit_footnote_reference(self, node: nodes.footnote_reference) -> None:
        number = node.astext().strip()
        docname = node['docname']
        if (docname, number) in self.appeared:
            footnote = self.appeared[docname, number]
            footnote['referred'] = True
            mark = footnotemark('', number, refid=node['refid'])
            node.replace_self(mark)
        else:
            footnote = self.get_footnote_by_reference(node)
            if self.restricted:
                mark = footnotemark('', number, refid=node['refid'])
                node.replace_self(mark)
                self.pendings.append(footnote)
            else:
                self.footnotes.remove(footnote)
                node.replace_self(footnote)
                footnote.walkabout(self)
            self.appeared[docname, number] = footnote
        raise nodes.SkipNode

    def get_footnote_by_reference(self, node: nodes.footnote_reference) -> nodes.footnote:
        docname = node['docname']
        for footnote in self.footnotes:
            if docname == footnote['docname'] and footnote['ids'][0] == node['refid']:
                return footnote
        raise ValueError('No footnote not found for given reference node %r' % node)