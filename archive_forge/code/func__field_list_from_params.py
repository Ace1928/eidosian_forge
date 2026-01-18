from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def _field_list_from_params(f: Union[Callable, TypeForm[Any]], cls: Optional[TypeForm[Any]], params: List[inspect.Parameter]) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    done = False
    while not done:
        done = True
        if hasattr(f, '__wrapped__'):
            f = f.__wrapped__
            done = False
        if isinstance(f, functools.partial):
            f = f.func
            done = False
    if inspect.isclass(f):
        cls = f
        f = f.__init__
    docstring = inspect.getdoc(f)
    docstring_from_arg_name = {}
    if docstring is not None:
        for param_doc in docstring_parser.parse(docstring).params:
            docstring_from_arg_name[param_doc.arg_name] = param_doc.description
    del docstring
    try:
        hints = _resolver.get_type_hints(f, include_extras=True)
    except TypeError:
        return UnsupportedNestedTypeMessage(f'Could not get hints for {f}!')
    field_list = []
    for param in params:
        default = param.default
        helptext = docstring_from_arg_name.get(param.name)
        if helptext is None and cls is not None:
            helptext = _docstrings.get_field_docstring(cls, param.name)
        if param.name not in hints:
            out = UnsupportedNestedTypeMessage(f"Expected fully type-annotated callable, but {f} with arguments {tuple(map(lambda p: p.name, params))} has no annotation for '{param.name}'.")
            if param.kind is param.KEYWORD_ONLY:
                raise _instantiators.UnsupportedTypeAnnotationError(out.message)
            return out
        markers: Tuple[Any, ...] = ()
        typ: Any = hints[param.name]
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            markers = (_markers.Positional, _markers._PositionalCall)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            markers = (_markers._UnpackArgsCall,)
            typ = Tuple.__getitem__((typ, ...))
            default = ()
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            markers = (_markers._UnpackKwargsCall,)
            typ = Dict.__getitem__((str, typ))
            default = {}
        field_list.append(FieldDefinition.make(name=param.name, type_or_callable=typ, default=default, is_default_from_default_instance=False, helptext=helptext, markers=markers))
    return field_list