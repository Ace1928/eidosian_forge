import posixpath
import re
import subprocess
from os import path
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.application import Sphinx
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, sha1
from sphinx.util.docutils import SphinxDirective, SphinxTranslator
from sphinx.util.fileutil import copy_asset
from sphinx.util.i18n import search_image_for_language
from sphinx.util.nodes import set_source_info
from sphinx.util.osutil import ensuredir
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.manpage import ManualPageTranslator
from sphinx.writers.texinfo import TexinfoTranslator
from sphinx.writers.text import TextTranslator
class Graphviz(SphinxDirective):
    """
    Directive to insert arbitrary dot markup.
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = False
    option_spec: OptionSpec = {'alt': directives.unchanged, 'align': align_spec, 'caption': directives.unchanged, 'layout': directives.unchanged, 'graphviz_dot': directives.unchanged, 'name': directives.unchanged, 'class': directives.class_option}

    def run(self) -> List[Node]:
        if self.arguments:
            document = self.state.document
            if self.content:
                return [document.reporter.warning(__('Graphviz directive cannot have both content and a filename argument'), line=self.lineno)]
            argument = search_image_for_language(self.arguments[0], self.env)
            rel_filename, filename = self.env.relfn2path(argument)
            self.env.note_dependency(rel_filename)
            try:
                with open(filename, encoding='utf-8') as fp:
                    dotcode = fp.read()
            except OSError:
                return [document.reporter.warning(__('External Graphviz file %r not found or reading it failed') % filename, line=self.lineno)]
        else:
            dotcode = '\n'.join(self.content)
            rel_filename = None
            if not dotcode.strip():
                return [self.state_machine.reporter.warning(__('Ignoring "graphviz" directive without content.'), line=self.lineno)]
        node = graphviz()
        node['code'] = dotcode
        node['options'] = {'docname': self.env.docname}
        if 'graphviz_dot' in self.options:
            node['options']['graphviz_dot'] = self.options['graphviz_dot']
        if 'layout' in self.options:
            node['options']['graphviz_dot'] = self.options['layout']
        if 'alt' in self.options:
            node['alt'] = self.options['alt']
        if 'align' in self.options:
            node['align'] = self.options['align']
        if 'class' in self.options:
            node['classes'] = self.options['class']
        if rel_filename:
            node['filename'] = rel_filename
        if 'caption' not in self.options:
            self.add_name(node)
            return [node]
        else:
            figure = figure_wrapper(self, node, self.options['caption'])
            self.add_name(figure)
            return [figure]