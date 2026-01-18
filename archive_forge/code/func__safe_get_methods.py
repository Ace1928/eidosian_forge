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
def _safe_get_methods(self, classes, name) -> Set[Callable]:
    return {method for class_ in classes for method in [getattr(class_, name, None)] if method}