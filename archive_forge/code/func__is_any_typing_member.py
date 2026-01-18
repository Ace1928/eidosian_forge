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
def _is_any_typing_member(node, scope_stack):
    """
    Determine whether `node` represents any member of a typing module.

    This is used as part of working out whether we are within a type annotation
    context.
    """
    return _is_typing_helper(node, lambda x: True, scope_stack)