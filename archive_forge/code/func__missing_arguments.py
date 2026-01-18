import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def _missing_arguments(f_name, argnames, pos, values):
    names = [repr(name) for name in argnames if name not in values]
    missing = len(names)
    if missing == 1:
        s = names[0]
    elif missing == 2:
        s = '{} and {}'.format(*names)
    else:
        tail = ', {} and {}'.format(*names[-2:])
        del names[-2:]
        s = ', '.join(names) + tail
    raise TypeError('%s() missing %i required %s argument%s: %s' % (f_name, missing, 'positional' if pos else 'keyword-only', '' if missing == 1 else 's', s))