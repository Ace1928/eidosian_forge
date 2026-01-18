from inspect import isclass, signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.decorators import undoc
def _create_duck_for_heap_type(duck_type):
    """Create an imitation of an object of a given type (a duck).

    Returns the duck or NOT_EVALUATED sentinel if duck could not be created.
    """
    duck = ImpersonatingDuck()
    try:
        duck.__class__ = duck_type
        return duck
    except TypeError:
        pass
    return NOT_EVALUATED