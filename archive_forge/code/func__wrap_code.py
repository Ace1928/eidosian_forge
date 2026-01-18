import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def _wrap_code(self, inner):
    yield (0, '<code>')
    yield from inner
    yield (0, '</code>')