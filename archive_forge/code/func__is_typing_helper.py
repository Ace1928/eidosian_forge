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
def _is_typing_helper(node, is_name_match_fn, scope_stack):
    """
    Internal helper to determine whether or not something is a member of a
    typing module. This is used as part of working out whether we are within a
    type annotation context.

    Note: you probably don't want to use this function directly. Instead see the
    utils below which wrap it (`_is_typing` and `_is_any_typing_member`).
    """

    def _bare_name_is_attr(name):
        for scope in reversed(scope_stack):
            if name in scope:
                return isinstance(scope[name], ImportationFrom) and scope[name].module in TYPING_MODULES and is_name_match_fn(scope[name].real_name)
        return False

    def _module_scope_is_typing(name):
        for scope in reversed(scope_stack):
            if name in scope:
                return isinstance(scope[name], Importation) and scope[name].fullName in TYPING_MODULES
        return False
    return isinstance(node, ast.Name) and _bare_name_is_attr(node.id) or (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and _module_scope_is_typing(node.value.id) and is_name_match_fn(node.attr))