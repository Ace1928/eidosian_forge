import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
class EnvVarXRefRole(XRefRole):
    """
    Cross-referencing role for environment variables (adds an index entry).
    """

    def result_nodes(self, document: nodes.document, env: 'BuildEnvironment', node: Element, is_ref: bool) -> Tuple[List[Node], List[system_message]]:
        if not is_ref:
            return ([node], [])
        varname = node['reftarget']
        tgtid = 'index-%s' % env.new_serialno('index')
        indexnode = addnodes.index()
        indexnode['entries'] = [('single', varname, tgtid, '', None), ('single', _('environment variable; %s') % varname, tgtid, '', None)]
        targetnode = nodes.target('', '', ids=[tgtid])
        document.note_explicit_target(targetnode)
        return ([indexnode, targetnode, node], [])