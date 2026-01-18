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
class TypeAliasModule:
    """Pseudo module class for autodoc_type_aliases."""

    def __init__(self, modname: str, mapping: Dict[str, str]) -> None:
        self.__modname = modname
        self.__mapping = mapping
        self.__module: Optional[ModuleType] = None

    def __getattr__(self, name: str) -> Any:
        fullname = '.'.join(filter(None, [self.__modname, name]))
        if fullname in self.__mapping:
            return TypeAliasForwardRef(self.__mapping[fullname])
        else:
            prefix = fullname + '.'
            nested = {k: v for k, v in self.__mapping.items() if k.startswith(prefix)}
            if nested:
                return TypeAliasModule(fullname, nested)
            else:
                try:
                    return import_module(fullname)
                except ImportError:
                    if self.__module is None:
                        self.__module = import_module(self.__modname)
                    return getattr(self.__module, name)