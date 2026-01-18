import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import docutils.parsers.rst.directives
import docutils.parsers.rst.roles
import docutils.parsers.rst.states
from docutils import nodes, utils
from docutils.nodes import Element, Node, TextElement, system_message
from sphinx import addnodes
from sphinx.locale import _, __
from sphinx.util import ws_re
from sphinx.util.docutils import ReferenceRole, SphinxRole
from sphinx.util.typing import RoleFunction
class AnyXRefRole(XRefRole):

    def process_link(self, env: 'BuildEnvironment', refnode: Element, has_explicit_title: bool, title: str, target: str) -> Tuple[str, str]:
        result = super().process_link(env, refnode, has_explicit_title, title, target)
        refnode.attributes.update(env.ref_context)
        return result