from __future__ import absolute_import
import functools
import json
import re
import sys
def __parser_input(self, token, string=None):
    self.lex_state = Parser.__lex_start
    self.buffer = ''
    self.parse_state(self, token, string)