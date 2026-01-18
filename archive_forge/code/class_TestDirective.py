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
class TestDirective(Directive):
    """This directive is useful only for testing purposes."""
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {'option': directives.unchanged_required}
    has_content = True

    def run(self):
        if self.content:
            text = '\n'.join(self.content)
            info = self.state_machine.reporter.info('Directive processed. Type="%s", arguments=%r, options=%r, content:' % (self.name, self.arguments, self.options), nodes.literal_block(text, text), line=self.lineno)
        else:
            info = self.state_machine.reporter.info('Directive processed. Type="%s", arguments=%r, options=%r, content: None' % (self.name, self.arguments, self.options), line=self.lineno)
        return [info]