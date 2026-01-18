import __future__
import difflib
import inspect
import linecache
import os
import pdb
import re
import sys
import traceback
import unittest
from io import StringIO, IncrementalNewlineDecoder
from collections import namedtuple
def _find_options(self, source, name, lineno):
    """
        Return a dictionary containing option overrides extracted from
        option directives in the given source string.

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.
        """
    options = {}
    for m in self._OPTION_DIRECTIVE_RE.finditer(source):
        option_strings = m.group(1).replace(',', ' ').split()
        for option in option_strings:
            if option[0] not in '+-' or option[1:] not in OPTIONFLAGS_BY_NAME:
                raise ValueError('line %r of the doctest for %s has an invalid option: %r' % (lineno + 1, name, option))
            flag = OPTIONFLAGS_BY_NAME[option[1:]]
            options[flag] = option[0] == '+'
    if options and self._IS_BLANK_OR_COMMENT(source):
        raise ValueError('line %r of the doctest for %s has an option directive on a line with no example: %r' % (lineno, name, source))
    return options