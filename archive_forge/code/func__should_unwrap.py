import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
def _should_unwrap(subject: Callable) -> bool:
    """Check the function should be unwrapped on getting signature."""
    __globals__ = getglobals(subject)
    if __globals__.get('__name__') == 'contextlib' and __globals__.get('__file__') == contextlib.__file__:
        return True
    return False