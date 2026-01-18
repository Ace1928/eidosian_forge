from __future__ import unicode_literals
import re
import warnings
from .. import util
from ..compat import unicode
from ..pattern import RegexPattern
@staticmethod
def _translate_segment_glob(pattern):
    """
		Translates the glob pattern to a regular expression. This is used in
		the constructor to translate a path segment glob pattern to its
		corresponding regular expression.

		*pattern* (:class:`str`) is the glob pattern.

		Returns the regular expression (:class:`str`).
		"""
    escape = False
    regex = ''
    i, end = (0, len(pattern))
    while i < end:
        char = pattern[i]
        i += 1
        if escape:
            escape = False
            regex += re.escape(char)
        elif char == '\\':
            escape = True
        elif char == '*':
            regex += '[^/]*'
        elif char == '?':
            regex += '[^/]'
        elif char == '[':
            j = i
            if j < end and pattern[j] == '!':
                j += 1
            if j < end and pattern[j] == ']':
                j += 1
            while j < end and pattern[j] != ']':
                j += 1
            if j < end:
                j += 1
                expr = '['
                if pattern[i] == '!':
                    expr += '^'
                    i += 1
                elif pattern[i] == '^':
                    expr += '\\^'
                    i += 1
                expr += pattern[i:j].replace('\\', '\\\\')
                regex += expr
                i = j
            else:
                regex += '\\['
        else:
            regex += re.escape(char)
    return regex