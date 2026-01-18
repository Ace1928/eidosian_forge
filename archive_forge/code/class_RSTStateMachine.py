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
class RSTStateMachine(StateMachineWS):
    """
    reStructuredText's master StateMachine.

    The entry point to reStructuredText parsing is the `run()` method.
    """

    def run(self, input_lines, document, input_offset=0, match_titles=True, inliner=None):
        """
        Parse `input_lines` and modify the `document` node in place.

        Extend `StateMachineWS.run()`: set up parse-global data and
        run the StateMachine.
        """
        self.language = languages.get_language(document.settings.language_code)
        self.match_titles = match_titles
        if inliner is None:
            inliner = Inliner()
        inliner.init_customizations(document.settings)
        self.memo = Struct(document=document, reporter=document.reporter, language=self.language, title_styles=[], section_level=0, section_bubble_up_kludge=False, inliner=inliner)
        self.document = document
        self.attach_observer(document.note_source)
        self.reporter = self.memo.reporter
        self.node = document
        results = StateMachineWS.run(self, input_lines, input_offset, input_source=document['source'])
        assert results == [], 'RSTStateMachine.run() results should be empty!'
        self.node = self.memo = None