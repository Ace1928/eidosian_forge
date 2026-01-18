from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_array_next(self, token, unused_string):
    if token == ',':
        self.parse_state = Parser.__parse_array_value
    elif token == ']':
        self.__parser_pop()
    else:
        self.__error("syntax error expecting ']' or ','")