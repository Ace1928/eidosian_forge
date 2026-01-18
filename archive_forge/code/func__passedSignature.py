from __future__ import annotations
import inspect
import sys
from dis import findlinestarts
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from warnings import warn, warn_explicit
from incremental import Version, getVersionString
from typing_extensions import ParamSpec
def _passedSignature(signature, positional, keyword):
    """
    Take an L{inspect.Signature}, a tuple of positional arguments, and a dict of
    keyword arguments, and return a mapping of arguments that were actually
    passed to their passed values.

    @param signature: The signature of the function to inspect.
    @type signature: L{inspect.Signature}

    @param positional: The positional arguments that were passed.
    @type positional: L{tuple}

    @param keyword: The keyword arguments that were passed.
    @type keyword: L{dict}

    @return: A dictionary mapping argument names (those declared in
        C{signature}) to values that were passed explicitly by the user.
    @rtype: L{dict} mapping L{str} to L{object}
    """
    result = {}
    kwargs = None
    numPositional = 0
    for n, (name, param) in enumerate(signature.parameters.items()):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            result[name] = positional[n:]
            numPositional = len(result[name]) + 1
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs = result[name] = {}
        elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
            if n < len(positional):
                result[name] = positional[n]
                numPositional += 1
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            if name not in keyword:
                if param.default == inspect.Parameter.empty:
                    raise TypeError(f'missing keyword arg {name}')
                else:
                    result[name] = param.default
        else:
            raise TypeError(f"'{name}' parameter is invalid kind: {param.kind}")
    if len(positional) > numPositional:
        raise TypeError('Too many arguments.')
    for name, value in keyword.items():
        if name in signature.parameters.keys():
            if name in result:
                raise TypeError('Already passed.')
            result[name] = value
        elif kwargs is not None:
            kwargs[name] = value
        else:
            raise TypeError('no such param')
    return result