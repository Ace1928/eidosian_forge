from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
class RegexSync(SyntaxSync):
    """
    Synchronize by starting at a line that matches the given regex pattern.
    """
    MAX_BACKWARDS = 500
    FROM_START_IF_NO_SYNC_POS_FOUND = 100

    def __init__(self, pattern):
        assert isinstance(pattern, six.text_type)
        self._compiled_pattern = re.compile(pattern)

    def get_sync_start_position(self, document, lineno):
        """ Scan backwards, and find a possible position to start. """
        pattern = self._compiled_pattern
        lines = document.lines
        for i in range(lineno, max(-1, lineno - self.MAX_BACKWARDS), -1):
            match = pattern.match(lines[i])
            if match:
                return (i, match.start())
        if lineno < self.FROM_START_IF_NO_SYNC_POS_FOUND:
            return (0, 0)
        else:
            return (lineno, 0)

    @classmethod
    def from_pygments_lexer_cls(cls, lexer_cls):
        """
        Create a :class:`.RegexSync` instance for this Pygments lexer class.
        """
        patterns = {'Python': '^\\s*(class|def)\\s+', 'Python 3': '^\\s*(class|def)\\s+', 'HTML': '<[/a-zA-Z]', 'JavaScript': '\\bfunction\\b'}
        p = patterns.get(lexer_cls.name, '^')
        return cls(p)