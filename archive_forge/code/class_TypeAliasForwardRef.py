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
class TypeAliasForwardRef:
    """Pseudo typing class for autodoc_type_aliases.

    This avoids the error on evaluating the type inside `get_type_hints()`.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self) -> None:
        pass

    def __eq__(self, other: Any) -> bool:
        return self.name == other

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name