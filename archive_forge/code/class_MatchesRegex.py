import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class MatchesRegex:
    """Matches if the matchee is matched by a regular expression."""

    def __init__(self, pattern, flags=0):
        self.pattern = pattern
        self.flags = flags

    def __str__(self):
        args = ['%r' % self.pattern]
        flag_arg = []
        for flag in dir(re):
            if len(flag) == 1:
                if self.flags & getattr(re, flag):
                    flag_arg.append('re.%s' % flag)
        if flag_arg:
            args.append('|'.join(flag_arg))
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))

    def match(self, value):
        if not re.match(self.pattern, value, self.flags):
            pattern = self.pattern
            if not isinstance(pattern, str):
                pattern = pattern.decode('latin1')
            pattern = pattern.encode('unicode_escape').decode('ascii')
            return Mismatch('{!r} does not match /{}/'.format(value, pattern.replace('\\\\', '\\')))