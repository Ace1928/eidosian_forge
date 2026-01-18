import sys
import os.path
import re
import time
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString, ErrorString
from docutils.utils.error_reporting import locale_encoding
from docutils.parsers.rst import Directive, convert_directive_function
from docutils.parsers.rst import directives, roles, states
from docutils.parsers.rst.directives.body import CodeBlock, NumberLines
from docutils.parsers.rst.roles import set_classes
from docutils.transforms import misc
class DefaultRole(Directive):
    """Set the default interpreted text role."""
    optional_arguments = 1
    final_argument_whitespace = False

    def run(self):
        if not self.arguments:
            if '' in roles._roles:
                del roles._roles['']
            return []
        role_name = self.arguments[0]
        role, messages = roles.role(role_name, self.state_machine.language, self.lineno, self.state.reporter)
        if role is None:
            error = self.state.reporter.error('Unknown interpreted text role "%s".' % role_name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            return messages + [error]
        roles._roles[''] = role
        return messages