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
class RFC2822List(SpecializedBody, RFC2822Body):
    """Second and subsequent RFC2822-style field_list fields."""
    patterns = RFC2822Body.patterns
    initial_transitions = RFC2822Body.initial_transitions

    def rfc2822(self, match, context, next_state):
        """RFC2822-style field list item."""
        field, blank_finish = self.rfc2822_field(match)
        self.parent += field
        self.blank_finish = blank_finish
        return ([], 'RFC2822List', [])
    blank = SpecializedBody.invalid_input