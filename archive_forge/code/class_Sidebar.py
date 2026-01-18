import sys
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.roles import set_classes
from docutils.utils.code_analyzer import Lexer, LexerError, NumberLines
class Sidebar(BasePseudoSection):
    node_class = nodes.sidebar
    option_spec = BasePseudoSection.option_spec.copy()
    option_spec['subtitle'] = directives.unchanged_required

    def run(self):
        if isinstance(self.state_machine.node, nodes.sidebar):
            raise self.error('The "%s" directive may not be used within a sidebar element.' % self.name)
        return BasePseudoSection.run(self)