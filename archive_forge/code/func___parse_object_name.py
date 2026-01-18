from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_object_name(self, token, string):
    if token == 'string':
        self.member_name = string
        self.parse_state = Parser.__parse_object_colon
    else:
        self.__error('syntax error parsing object expecting string')