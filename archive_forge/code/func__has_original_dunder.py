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
def _has_original_dunder(value, allowed_types, allowed_methods, allowed_external, method_name):
    value_type = type(value)
    if value_type in allowed_types:
        return True
    method = getattr(value_type, method_name, None)
    if method is None:
        return None
    if method in allowed_methods:
        return True
    for module_name, *access_path in allowed_external:
        if _has_original_dunder_external(value, module_name, access_path, method_name):
            return True
    return False