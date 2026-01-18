from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.util import debug_output
class Patterns(object):
    """Holds multiple regex strings and checks matches against all."""

    def __init__(self, pattern_strings, ignore_prefix_length=0):
        """Initializes class."""
        self._patterns = [re.compile(x) for x in pattern_strings]
        self._ignore_prefix_length = ignore_prefix_length

    def match(self, target):
        """Checks if string matches any stored pattern."""
        target_substring = target[self._ignore_prefix_length:]
        return any((p.match(target_substring) for p in self._patterns))

    def __repr__(self):
        return debug_output.generic_repr(self)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._patterns == other._patterns and self._ignore_prefix_length == other._ignore_prefix_length