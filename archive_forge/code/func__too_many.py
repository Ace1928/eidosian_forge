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
def _too_many(f_name, args, kwonly, varargs, defcount, given, values):
    atleast = len(args) - defcount
    kwonly_given = len([arg for arg in kwonly if arg in values])
    if varargs:
        plural = atleast != 1
        sig = 'at least %d' % (atleast,)
    elif defcount:
        plural = True
        sig = 'from %d to %d' % (atleast, len(args))
    else:
        plural = len(args) != 1
        sig = str(len(args))
    kwonly_sig = ''
    if kwonly_given:
        msg = ' positional argument%s (and %d keyword-only argument%s)'
        kwonly_sig = msg % ('s' if given != 1 else '', kwonly_given, 's' if kwonly_given != 1 else '')
    raise TypeError('%s() takes %s positional argument%s but %d%s %s given' % (f_name, sig, 's' if plural else '', given, kwonly_sig, 'was' if given == 1 and (not kwonly_given) else 'were'))