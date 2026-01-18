from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
def _has_original_dunder_external(value, module_name: str, access_path: Sequence[str], method_name: str):
    if module_name not in sys.modules:
        return False
    try:
        member_type = _get_external(module_name, access_path)
        value_type = type(value)
        if type(value) == member_type:
            return True
        if method_name == '__getattribute__':
            return False
        if isinstance(value, member_type):
            method = getattr(value_type, method_name, None)
            member_method = getattr(member_type, method_name, None)
            if member_method == method:
                return True
    except (AttributeError, KeyError):
        return False