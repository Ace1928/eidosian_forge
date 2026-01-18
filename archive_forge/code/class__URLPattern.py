import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
class _URLPattern(object):
    """Internal class which represents a URL pattern."""

    def __init__(self, pattern):
        self._pattern = pattern
        self.priority = len(pattern)
        self._contains_asterisk = '*' in self._pattern
        self._contains_dollar = self._pattern.endswith('$')
        if self._contains_asterisk:
            self._pattern_before_asterisk = self._pattern[:self._pattern.find('*')]
        elif self._contains_dollar:
            self._pattern_before_dollar = self._pattern[:-1]
        self._pattern_compiled = False

    def match(self, url):
        """Return True if pattern matches the given URL, otherwise return False."""
        if self._pattern_compiled:
            return self._pattern.match(url)
        if not self._contains_asterisk:
            if not self._contains_dollar:
                return url.startswith(self._pattern)
            return url == self._pattern_before_dollar
        if not url.startswith(self._pattern_before_asterisk):
            return False
        self._pattern = self._prepare_pattern_for_regex(self._pattern)
        self._pattern = re.compile(self._pattern)
        self._pattern_compiled = True
        return self._pattern.match(url)

    def _prepare_pattern_for_regex(self, pattern):
        """Return equivalent regex pattern for the given URL pattern."""
        pattern = re.sub('\\*+', '*', pattern)
        s = re.split('(\\*|\\$$)', pattern)
        for index, substr in enumerate(s):
            if substr not in _WILDCARDS:
                s[index] = re.escape(substr)
            elif s[index] == '*':
                s[index] = '.*?'
        pattern = ''.join(s)
        return pattern