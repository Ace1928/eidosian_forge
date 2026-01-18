import builtins
import inspect
import re
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.graphviz import (figure_wrapper, graphviz, render_dot_html, render_dot_latex,
from sphinx.util import md5
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator
class InheritanceDiagram(SphinxDirective):
    """
    Run when the inheritance_diagram directive is first encountered.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {'parts': int, 'private-bases': directives.flag, 'caption': directives.unchanged, 'top-classes': directives.unchanged_required}

    def run(self) -> List[Node]:
        node = inheritance_diagram()
        node.document = self.state.document
        class_names = self.arguments[0].split()
        class_role = self.env.get_domain('py').role('class')
        node['parts'] = self.options.get('parts', 0)
        node['content'] = ', '.join(class_names)
        node['top-classes'] = []
        for cls in self.options.get('top-classes', '').split(','):
            cls = cls.strip()
            if cls:
                node['top-classes'].append(cls)
        try:
            graph = InheritanceGraph(class_names, self.env.ref_context.get('py:module'), parts=node['parts'], private_bases='private-bases' in self.options, aliases=self.config.inheritance_alias, top_classes=node['top-classes'])
        except InheritanceException as err:
            return [node.document.reporter.warning(err, line=self.lineno)]
        for name in graph.get_all_class_names():
            refnodes, x = class_role('class', ':class:`%s`' % name, name, 0, self.state)
            node.extend(refnodes)
        node['graph'] = graph
        if 'caption' not in self.options:
            self.add_name(node)
            return [node]
        else:
            figure = figure_wrapper(self, node, self.options['caption'])
            self.add_name(figure)
            return [figure]