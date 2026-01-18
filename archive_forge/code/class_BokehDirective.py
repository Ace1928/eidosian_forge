from __future__ import annotations
import logging  # isort:skip
import re
from docutils import nodes
from docutils.statemachine import ViewList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
class BokehDirective(SphinxDirective):

    def parse(self, rst_text, annotation):
        result = ViewList()
        for line in rst_text.split('\n'):
            result.append(line, annotation)
        node = nodes.paragraph()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        return node.children