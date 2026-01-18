import sys
import os
class _TrivialRe:

    def __init__(self, *patterns):
        self._patterns = patterns

    def match(self, string):
        return all((pat in string for pat in self._patterns))