from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_start(self, token, unused_string):
    if token == '{':
        self.__push_object()
    elif token == '[':
        self.__push_array()
    else:
        self.__error('syntax error at beginning of input')