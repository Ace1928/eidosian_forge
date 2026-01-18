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
class SpecializedBody(Body):
    """
    Superclass for second and subsequent compound element members.  Compound
    elements are lists and list-like constructs.

    All transition methods are disabled (redefined as `invalid_input`).
    Override individual methods in subclasses to re-enable.

    For example, once an initial bullet list item, say, is recognized, the
    `BulletList` subclass takes over, with a "bullet_list" node as its
    container.  Upon encountering the initial bullet list item, `Body.bullet`
    calls its ``self.nested_list_parse`` (`RSTState.nested_list_parse`), which
    starts up a nested parsing session with `BulletList` as the initial state.
    Only the ``bullet`` transition method is enabled in `BulletList`; as long
    as only bullet list items are encountered, they are parsed and inserted
    into the container.  The first construct which is *not* a bullet list item
    triggers the `invalid_input` method, which ends the nested parse and
    closes the container.  `BulletList` needs to recognize input that is
    invalid in the context of a bullet list, which means everything *other
    than* bullet list items, so it inherits the transition list created in
    `Body`.
    """

    def invalid_input(self, match=None, context=None, next_state=None):
        """Not a compound element member. Abort this state machine."""
        self.state_machine.previous_line()
        raise EOFError
    indent = invalid_input
    bullet = invalid_input
    enumerator = invalid_input
    field_marker = invalid_input
    option_marker = invalid_input
    doctest = invalid_input
    line_block = invalid_input
    grid_table_top = invalid_input
    simple_table_top = invalid_input
    explicit_markup = invalid_input
    anonymous = invalid_input
    line = invalid_input
    text = invalid_input