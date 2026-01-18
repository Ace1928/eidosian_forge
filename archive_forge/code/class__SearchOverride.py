import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
class _SearchOverride:
    """
    Mix-in class to override `StateMachine` regular expression behavior.

    Changes regular expression matching, from the default `re.match()`
    (succeeds only if the pattern matches at the start of `self.line`) to
    `re.search()` (succeeds if the pattern matches anywhere in `self.line`).
    When subclassing a `StateMachine`, list this class **first** in the
    inheritance list of the class definition.
    """

    def match(self, pattern):
        """
        Return the result of a regular expression search.

        Overrides `StateMachine.match()`.

        Parameter `pattern`: `re` compiled regular expression.
        """
        return pattern.search(self.line)