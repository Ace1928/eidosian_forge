from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_object_init(self, token, string):
    if token == '}':
        self.__parser_pop()
    else:
        self.__parse_object_name(token, string)