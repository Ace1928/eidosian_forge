from typing import Any, Dict, List, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.util import logging, texescape
from sphinx.util.docutils import SphinxDirective, new_document
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
class TodoDomain(Domain):
    name = 'todo'
    label = 'todo'

    @property
    def todos(self) -> Dict[str, List[todo_node]]:
        return self.data.setdefault('todos', {})

    def clear_doc(self, docname: str) -> None:
        self.todos.pop(docname, None)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for docname in docnames:
            self.todos[docname] = otherdata['todos'][docname]

    def process_doc(self, env: BuildEnvironment, docname: str, document: nodes.document) -> None:
        todos = self.todos.setdefault(docname, [])
        for todo in document.findall(todo_node):
            env.app.emit('todo-defined', todo)
            todos.append(todo)
            if env.config.todo_emit_warnings:
                logger.warning(__('TODO entry found: %s'), todo[1].astext(), location=todo)