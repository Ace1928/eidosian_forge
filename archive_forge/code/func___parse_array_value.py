from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_array_value(self, token, string):
    self.__parse_value(token, string, Parser.__parse_array_next)