import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
def _get_debug_error_message(module, old_lines, new_lines):
    current_lines = split_lines(module.get_code(), keepends=True)
    current_diff = difflib.unified_diff(new_lines, current_lines)
    old_new_diff = difflib.unified_diff(old_lines, new_lines)
    import parso
    return "There's an issue with the diff parser. Please report (parso v%s) - Old/New:\n%s\nActual Diff (May be empty):\n%s" % (parso.__version__, ''.join(old_new_diff), ''.join(current_diff))