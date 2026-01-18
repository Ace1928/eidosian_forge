import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
class QuotedLiteralBlock(RSTState):
    """
    Nested parse handler for quoted (unindented) literal blocks.

    Special-purpose.  Not for inclusion in `state_classes`.
    """
    patterns = {'initial_quoted': '(%(nonalphanum7bit)s)' % Body.pats, 'text': ''}
    initial_transitions = ('initial_quoted', 'text')

    def __init__(self, state_machine, debug=False):
        RSTState.__init__(self, state_machine, debug)
        self.messages = []
        self.initial_lineno = None

    def blank(self, match, context, next_state):
        if context:
            raise EOFError
        else:
            return (context, next_state, [])

    def eof(self, context):
        if context:
            src, srcline = self.state_machine.get_source_and_line(self.initial_lineno)
            text = '\n'.join(context)
            literal_block = nodes.literal_block(text, text)
            literal_block.source = src
            literal_block.line = srcline
            self.parent += literal_block
        else:
            self.parent += self.reporter.warning('Literal block expected; none found.', line=self.state_machine.abs_line_number())
            self.state_machine.previous_line()
        self.parent += self.messages
        return []

    def indent(self, match, context, next_state):
        assert context, 'QuotedLiteralBlock.indent: context should not be empty!'
        self.messages.append(self.reporter.error('Unexpected indentation.', line=self.state_machine.abs_line_number()))
        self.state_machine.previous_line()
        raise EOFError

    def initial_quoted(self, match, context, next_state):
        """Match arbitrary quote character on the first line only."""
        self.remove_transition('initial_quoted')
        quote = match.string[0]
        pattern = re.compile(re.escape(quote), re.UNICODE)
        self.add_transition('quoted', (pattern, self.quoted, self.__class__.__name__))
        self.initial_lineno = self.state_machine.abs_line_number()
        return ([match.string], next_state, [])

    def quoted(self, match, context, next_state):
        """Match consistent quotes on subsequent lines."""
        context.append(match.string)
        return (context, next_state, [])

    def text(self, match, context, next_state):
        if context:
            self.messages.append(self.reporter.error('Inconsistent literal block quoting.', line=self.state_machine.abs_line_number()))
            self.state_machine.previous_line()
        raise EOFError