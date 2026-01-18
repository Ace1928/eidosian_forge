import sys
from typing import Any, Dict, List, NamedTuple
from docutils import nodes
from docutils.nodes import Node, TextElement
from pygments.lexers import PythonConsoleLexer, guess_lexer
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import doctest
from sphinx.transforms import SphinxTransform
class HighlightLanguageVisitor(nodes.NodeVisitor):

    def __init__(self, document: nodes.document, default_language: str) -> None:
        self.default_setting = HighlightSetting(default_language, False, sys.maxsize)
        self.settings: List[HighlightSetting] = []
        super().__init__(document)

    def unknown_visit(self, node: Node) -> None:
        pass

    def unknown_departure(self, node: Node) -> None:
        pass

    def visit_document(self, node: Node) -> None:
        self.settings.append(self.default_setting)

    def depart_document(self, node: Node) -> None:
        self.settings.pop()

    def visit_start_of_file(self, node: Node) -> None:
        self.settings.append(self.default_setting)

    def depart_start_of_file(self, node: Node) -> None:
        self.settings.pop()

    def visit_highlightlang(self, node: addnodes.highlightlang) -> None:
        self.settings[-1] = HighlightSetting(node['lang'], node['force'], node['linenothreshold'])

    def visit_literal_block(self, node: nodes.literal_block) -> None:
        setting = self.settings[-1]
        if 'language' not in node:
            node['language'] = setting.language
            node['force'] = setting.force
        if 'linenos' not in node:
            lines = node.astext().count('\n')
            node['linenos'] = lines >= setting.lineno_threshold - 1