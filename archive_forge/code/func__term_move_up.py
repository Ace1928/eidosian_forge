import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _term_move_up():
    return '' if os.name == 'nt' and colorama is None else '\x1b[A'