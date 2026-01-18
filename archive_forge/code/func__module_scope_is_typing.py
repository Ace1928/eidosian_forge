import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def _module_scope_is_typing(name):
    for scope in reversed(scope_stack):
        if name in scope:
            return isinstance(scope[name], Importation) and scope[name].fullName in TYPING_MODULES
    return False