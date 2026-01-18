from __future__ import absolute_import
import functools
import json
import re
import sys
def __parser_push(self, new_json, next_state):
    if len(self.stack) < Parser.MAX_HEIGHT:
        if len(self.stack) > 0:
            self.__put_value(new_json)
        self.stack.append(new_json)
        self.parse_state = next_state
    else:
        self.__error('input exceeds maximum nesting depth %d' % Parser.MAX_HEIGHT)