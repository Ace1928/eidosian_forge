from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
class _ExprHAS(_ExprWordMatchBase):
    """HAS word match node."""

    def __init__(self, backend, key, operand, transform):
        super(_ExprHAS, self).__init__(backend, key, operand, transform, op=':', warned_attribute='_deprecated_has_warned')

    def _AddPattern(self, pattern):
        """Adds a HAS match pattern to self._patterns.

    A pattern is a word that optionally contains one trailing * that matches
    0 or more characters.

    This method re-implements both the original and the OnePlatform : using REs.
    It was tested against the original tests with no failures.  This cleaned up
    the code (really!) and made it easier to reason about the two
    implementations.

    Args:
      pattern: A string containing at most one trailing *.

    Raises:
      resource_exceptions.ExpressionSyntaxError if the pattern contains more
        than one leading or trailing * glob character.
    """
        if pattern == '*':
            standard_pattern = '.'
            deprecated_pattern = None
        else:
            head = '\\b'
            glob = ''
            tail = '\\b'
            normalized_pattern = NormalizeForSearch(pattern)
            parts = normalized_pattern.split('*')
            if len(parts) > 2:
                raise resource_exceptions.ExpressionSyntaxError('At most one * expected in : patterns [{}].'.format(pattern))
            if normalized_pattern.endswith('*'):
                normalized_pattern = normalized_pattern[:-1]
                tail = ''
            word = re.escape(normalized_pattern)
            standard_pattern = head + word + tail
            if len(parts) == 1:
                parts.append('')
            elif pattern.startswith('*'):
                head = ''
            elif pattern.endswith('*'):
                tail = ''
            else:
                glob = '.*'
            left = re.escape(parts[0]) if parts[0] else ''
            right = re.escape(parts[1]) if parts[1] else ''
            if head and tail:
                if glob:
                    deprecated_pattern = '^' + left + glob + right + '$'
                else:
                    deprecated_pattern = left + glob + right
            elif head:
                deprecated_pattern = '^' + left + glob + right
            elif tail:
                deprecated_pattern = left + glob + right + '$'
            else:
                deprecated_pattern = None
        reflags = re.IGNORECASE | re.MULTILINE | re.UNICODE
        standard_regex = _ReCompile(standard_pattern, reflags)
        if deprecated_pattern:
            deprecated_regex = _ReCompile(deprecated_pattern, reflags)
        else:
            deprecated_regex = None
        self._patterns.append((pattern, standard_regex, deprecated_regex))