from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
class DetailsPrinter:

    def __init__(self, target):
        self._target = target
        self._parenthesed = False

    def append(self, what):
        if not self._parenthesed:
            self._target += ' ('
            self._parenthesed = True
        else:
            self._target += ', '
        self._target += what

    def finish(self):
        if self._parenthesed:
            self._target += ')'
        return self._target