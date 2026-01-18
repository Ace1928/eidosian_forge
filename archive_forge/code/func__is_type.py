import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
def _is_type(annotation, cls, a_module, a_type, is_type_predicate):
    match = _MODULE_IDENTIFIER_RE.match(annotation)
    if match:
        ns = None
        module_name = match.group(1)
        if not module_name:
            ns = sys.modules.get(cls.__module__).__dict__
        else:
            module = sys.modules.get(cls.__module__)
            if module and module.__dict__.get(module_name) is a_module:
                ns = sys.modules.get(a_type.__module__).__dict__
        if ns and is_type_predicate(ns.get(match.group(2)), a_module):
            return True
    return False