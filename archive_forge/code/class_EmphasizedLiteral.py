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
class EmphasizedLiteral(SphinxRole):
    parens_re = re.compile('(\\\\\\\\|\\\\{|\\\\}|{|})')

    def run(self) -> Tuple[List[Node], List[system_message]]:
        children = self.parse(self.text)
        node = nodes.literal(self.rawtext, '', *children, role=self.name.lower(), classes=[self.name])
        return ([node], [])

    def parse(self, text: str) -> List[Node]:
        result: List[Node] = []
        stack = ['']
        for part in self.parens_re.split(text):
            if part == '\\\\':
                stack[-1] += '\\'
            elif part == '{':
                if len(stack) >= 2 and stack[-2] == '{':
                    stack[-1] += '{'
                else:
                    stack.append('{')
                    stack.append('')
            elif part == '}':
                if len(stack) == 3 and stack[1] == '{' and (len(stack[2]) > 0):
                    if stack[0]:
                        result.append(nodes.Text(stack[0]))
                    result.append(nodes.emphasis(stack[2], stack[2]))
                    stack = ['']
                else:
                    stack.append('}')
                    stack = [''.join(stack)]
            elif part == '\\{':
                stack[-1] += '{'
            elif part == '\\}':
                stack[-1] += '}'
            else:
                stack[-1] += part
        if ''.join(stack):
            text = ''.join(stack)
            result.append(nodes.Text(text))
        return result