import inspect
import keyword
import pydoc
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Dict, List, ContextManager
from types import MemberDescriptorType, TracebackType
from ._typing_compat import Literal
from pygments.token import Token
from pygments.lexers import Python3Lexer
from .lazyre import LazyReCompile
def _get_argspec_from_signature(f: Callable) -> ArgSpec:
    """Get callable signature from inspect.signature in argspec format.

    inspect.signature is a Python 3 only function that returns the signature of
    a function.  Its advantage over inspect.getfullargspec is that it returns
    the signature of a decorated function, if the wrapper function itself is
    decorated with functools.wraps.

    """
    args = []
    varargs = None
    varkwargs = None
    defaults = []
    kwonly = []
    kwonly_defaults = {}
    annotations = {}
    signature = inspect.signature(f)
    for parameter in signature.parameters.values():
        if parameter.annotation is not parameter.empty:
            annotations[parameter.name] = parameter.annotation
        if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            args.append(parameter.name)
            if parameter.default is not parameter.empty:
                defaults.append(parameter.default)
        elif parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            args.append(parameter.name)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            varargs = parameter.name
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwonly.append(parameter.name)
            kwonly_defaults[parameter.name] = parameter.default
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            varkwargs = parameter.name
    return ArgSpec(args, varargs, varkwargs, defaults if defaults else None, kwonly, kwonly_defaults if kwonly_defaults else None, annotations if annotations else None)