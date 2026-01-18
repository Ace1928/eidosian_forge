from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_object_next(self, token, unused_string):
    if token == ',':
        self.parse_state = Parser.__parse_object_name
    elif token == '}':
        self.__parser_pop()
    else:
        self.__error("syntax error expecting '}' or ','")