import sys
import textwrap
from difflib import unified_diff
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.config import Config
from sphinx.directives import optional_int
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
class LiteralInclude(SphinxDirective):
    """
    Like ``.. include:: :literal:``, but only warns if the include file is
    not found, and does not raise errors.  Also has several options for
    selecting what to include.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {'dedent': optional_int, 'linenos': directives.flag, 'lineno-start': int, 'lineno-match': directives.flag, 'tab-width': int, 'language': directives.unchanged_required, 'force': directives.flag, 'encoding': directives.encoding, 'pyobject': directives.unchanged_required, 'lines': directives.unchanged_required, 'start-after': directives.unchanged_required, 'end-before': directives.unchanged_required, 'start-at': directives.unchanged_required, 'end-at': directives.unchanged_required, 'prepend': directives.unchanged_required, 'append': directives.unchanged_required, 'emphasize-lines': directives.unchanged_required, 'caption': directives.unchanged, 'class': directives.class_option, 'name': directives.unchanged, 'diff': directives.unchanged_required}

    def run(self) -> List[Node]:
        document = self.state.document
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled', line=self.lineno)]
        if 'diff' in self.options:
            _, path = self.env.relfn2path(self.options['diff'])
            self.options['diff'] = path
        try:
            location = self.state_machine.get_source_and_line(self.lineno)
            rel_filename, filename = self.env.relfn2path(self.arguments[0])
            self.env.note_dependency(rel_filename)
            reader = LiteralIncludeReader(filename, self.options, self.config)
            text, lines = reader.read(location=location)
            retnode: Element = nodes.literal_block(text, text, source=filename)
            retnode['force'] = 'force' in self.options
            self.set_source_info(retnode)
            if self.options.get('diff'):
                retnode['language'] = 'udiff'
            elif 'language' in self.options:
                retnode['language'] = self.options['language']
            if 'linenos' in self.options or 'lineno-start' in self.options or 'lineno-match' in self.options:
                retnode['linenos'] = True
            retnode['classes'] += self.options.get('class', [])
            extra_args = retnode['highlight_args'] = {}
            if 'emphasize-lines' in self.options:
                hl_lines = parselinenos(self.options['emphasize-lines'], lines)
                if any((i >= lines for i in hl_lines)):
                    logger.warning(__('line number spec is out of range(1-%d): %r') % (lines, self.options['emphasize-lines']), location=location)
                extra_args['hl_lines'] = [x + 1 for x in hl_lines if x < lines]
            extra_args['linenostart'] = reader.lineno_start
            if 'caption' in self.options:
                caption = self.options['caption'] or self.arguments[0]
                retnode = container_wrapper(self, retnode, caption)
            self.add_name(retnode)
            return [retnode]
        except Exception as exc:
            return [document.reporter.warning(exc, line=self.lineno)]