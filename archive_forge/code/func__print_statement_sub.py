import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
def _print_statement_sub(match):
    expr = match.groups('expr')
    return 'print(%s)' % expr