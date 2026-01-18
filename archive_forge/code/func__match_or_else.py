import string
import warnings
from json import loads
from jmespath.exceptions import LexerError, EmptyExpressionError
def _match_or_else(self, expected, match_type, else_type):
    start = self._position
    current = self._current
    next_char = self._next()
    if next_char == expected:
        self._next()
        return {'type': match_type, 'value': current + next_char, 'start': start, 'end': start + 1}
    return {'type': else_type, 'value': current, 'start': start, 'end': start}