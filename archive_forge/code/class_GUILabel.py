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
class GUILabel(SphinxRole):
    amp_re = re.compile('(?<!&)&(?![&\\s])')

    def run(self) -> Tuple[List[Node], List[system_message]]:
        node = nodes.inline(rawtext=self.rawtext, classes=[self.name])
        spans = self.amp_re.split(self.text)
        node += nodes.Text(spans.pop(0))
        for span in spans:
            span = span.replace('&&', '&')
            letter = nodes.Text(span[0])
            accelerator = nodes.inline('', '', letter, classes=['accelerator'])
            node += accelerator
            node += nodes.Text(span[1:])
        return ([node], [])