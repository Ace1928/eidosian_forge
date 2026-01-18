from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.domains.python import _pseudo_parse_arglist
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import make_id, make_refnode, nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class JSModule(SphinxDirective):
    """
    Directive to mark description of a new JavaScript module.

    This directive specifies the module name that will be used by objects that
    follow this directive.

    Options
    -------

    noindex
        If the ``noindex`` option is specified, no linkable elements will be
        created, and the module won't be added to the global module index. This
        is useful for splitting up the module definition across multiple
        sections or files.

    :param mod_name: Module name
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec: OptionSpec = {'noindex': directives.flag, 'nocontentsentry': directives.flag}

    def run(self) -> List[Node]:
        mod_name = self.arguments[0].strip()
        self.env.ref_context['js:module'] = mod_name
        noindex = 'noindex' in self.options
        content_node: Element = nodes.section()
        with switch_source_input(self.state, self.content):
            content_node.document = self.state.document
            nested_parse_with_titles(self.state, self.content, content_node)
        ret: List[Node] = []
        if not noindex:
            domain = cast(JavaScriptDomain, self.env.get_domain('js'))
            node_id = make_id(self.env, self.state.document, 'module', mod_name)
            domain.note_module(mod_name, node_id)
            domain.note_object(mod_name, 'module', node_id, location=(self.env.docname, self.lineno))
            target = nodes.target('', '', ids=[node_id], ismod=True)
            self.state.document.note_explicit_target(target)
            ret.append(target)
            indextext = _('%s (module)') % mod_name
            inode = addnodes.index(entries=[('single', indextext, node_id, '', None)])
            ret.append(inode)
        ret.extend(content_node.children)
        return ret

    def make_old_id(self, modname: str) -> str:
        """Generate old styled node_id for JS modules.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return 'module-' + modname