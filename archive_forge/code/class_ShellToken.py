from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class ShellToken(object):
    """Shell token info.

  Attributes:
    value: The token string with quotes and escapes preserved.
    lex: Instance of ShellTokenType
    start: the index of the first char of the raw value
    end: the index directly after the last char of the raw value
  """

    def __init__(self, value, lex=ShellTokenType.ARG, start=None, end=None):
        self.value = value
        self.lex = lex
        self.start = start
        self.end = end

    def UnquotedValue(self):
        if self.lex is ShellTokenType.ARG or self.lex is ShellTokenType.FLAG:
            return UnquoteShell(self.value)
        else:
            return self.value

    def __eq__(self, other):
        """Equality based on properties."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        """Improve debugging during tests."""
        return 'ShellToken({}, {}, {}, {})'.format(self.value, self.lex, self.start, self.end)