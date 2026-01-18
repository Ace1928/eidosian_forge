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
def _signature_strip_non_python_syntax(signature):
    """
    Private helper function. Takes a signature in Argument Clinic's
    extended signature format.

    Returns a tuple of three things:
      * that signature re-rendered in standard Python syntax,
      * the index of the "self" parameter (generally 0), or None if
        the function does not have a "self" parameter, and
      * the index of the last "positional only" parameter,
        or None if the signature has no positional-only parameters.
    """
    if not signature:
        return (signature, None, None)
    self_parameter = None
    last_positional_only = None
    lines = [l.encode('ascii') for l in signature.split('\n') if l]
    generator = iter(lines).__next__
    token_stream = tokenize.tokenize(generator)
    delayed_comma = False
    skip_next_comma = False
    text = []
    add = text.append
    current_parameter = 0
    OP = token.OP
    ERRORTOKEN = token.ERRORTOKEN
    t = next(token_stream)
    assert t.type == tokenize.ENCODING
    for t in token_stream:
        type, string = (t.type, t.string)
        if type == OP:
            if string == ',':
                if skip_next_comma:
                    skip_next_comma = False
                else:
                    assert not delayed_comma
                    delayed_comma = True
                    current_parameter += 1
                continue
            if string == '/':
                assert not skip_next_comma
                assert last_positional_only is None
                skip_next_comma = True
                last_positional_only = current_parameter - 1
                continue
        if type == ERRORTOKEN and string == '$':
            assert self_parameter is None
            self_parameter = current_parameter
            continue
        if delayed_comma:
            delayed_comma = False
            if not (type == OP and string == ')'):
                add(', ')
        add(string)
        if string == ',':
            add(' ')
    clean_signature = ''.join(text)
    return (clean_signature, self_parameter, last_positional_only)