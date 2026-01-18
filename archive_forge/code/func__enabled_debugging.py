import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _enabled_debugging(self, old_lines, lines_new):
    if self._module.get_code() != ''.join(lines_new):
        LOG.warning('parser issue:\n%s\n%s', ''.join(old_lines), ''.join(lines_new))