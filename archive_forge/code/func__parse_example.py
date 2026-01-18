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
def _parse_example(self, m, name, lineno):
    """
        Given a regular expression match from `_EXAMPLE_RE` (`m`),
        return a pair `(source, want)`, where `source` is the matched
        example's source code (with prompts and indentation stripped);
        and `want` is the example's expected output (with indentation
        stripped).

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.
        """
    indent = len(m.group('indent'))
    source_lines = m.group('source').split('\n')
    self._check_prompt_blank(source_lines, indent, name, lineno)
    self._check_prefix(source_lines[1:], ' ' * indent + '.', name, lineno)
    source = '\n'.join([sl[indent + 4:] for sl in source_lines])
    want = m.group('want')
    want_lines = want.split('\n')
    if len(want_lines) > 1 and re.match(' *$', want_lines[-1]):
        del want_lines[-1]
    self._check_prefix(want_lines, ' ' * indent, name, lineno + len(source_lines))
    want = '\n'.join([wl[indent:] for wl in want_lines])
    m = self._EXCEPTION_RE.match(want)
    if m:
        exc_msg = m.group('msg')
    else:
        exc_msg = None
    options = self._find_options(source, name, lineno)
    return (source, options, want, exc_msg)