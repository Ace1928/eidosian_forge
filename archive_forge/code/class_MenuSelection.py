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
class MenuSelection(GUILabel):
    BULLET_CHARACTER = '‣'

    def run(self) -> Tuple[List[Node], List[system_message]]:
        self.text = self.text.replace('-->', self.BULLET_CHARACTER)
        return super().run()