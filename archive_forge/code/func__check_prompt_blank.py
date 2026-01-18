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
def _check_prompt_blank(self, lines, indent, name, lineno):
    """
        Given the lines of a source string (including prompts and
        leading indentation), check to make sure that every prompt is
        followed by a space character.  If any line is not followed by
        a space character, then raise ValueError.
        """
    for i, line in enumerate(lines):
        if len(line) >= indent + 4 and line[indent + 3] != ' ':
            raise ValueError('line %r of the docstring for %s lacks blank after %s: %r' % (lineno + i + 1, name, line[indent:indent + 3], line))