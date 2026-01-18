from __future__ import absolute_import
import functools
import json
import re
import sys
def __parser_pop(self):
    if len(self.stack) == 1:
        self.parse_state = Parser.__parse_end
        if not self.check_trailer:
            self.done = True
    else:
        self.stack.pop()
        top = self.stack[-1]
        if isinstance(top, list):
            self.parse_state = Parser.__parse_array_next
        else:
            self.parse_state = Parser.__parse_object_next