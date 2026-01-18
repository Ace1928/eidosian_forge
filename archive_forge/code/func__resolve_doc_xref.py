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
def _resolve_doc_xref(self, env: 'BuildEnvironment', fromdocname: str, builder: 'Builder', typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
    refdoc = node.get('refdoc', fromdocname)
    docname = docname_join(refdoc, node['reftarget'])
    if docname not in env.all_docs:
        return None
    else:
        if node['refexplicit']:
            caption = node.astext()
        else:
            caption = clean_astext(env.titles[docname])
        innernode = nodes.inline(caption, caption, classes=['doc'])
        return make_refnode(builder, fromdocname, docname, None, innernode)