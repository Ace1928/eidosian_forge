import warnings
from collections import ChainMap
from functools import partial, partialmethod, wraps
from itertools import chain
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, overload
from .errors import ConfigError
from .typing import AnyCallable
from .utils import ROOT_KEY, in_ipython
def _prepare_validator(function: AnyCallable, allow_reuse: bool) -> 'AnyClassMethod':
    """
    Avoid validators with duplicated names since without this, validators can be overwritten silently
    which generally isn't the intended behaviour, don't run in ipython (see #312) or if allow_reuse is False.
    """
    f_cls = function if isinstance(function, classmethod) else classmethod(function)
    if not in_ipython() and (not allow_reuse):
        ref = getattr(f_cls.__func__, '__module__', '<No __module__>') + '.' + getattr(f_cls.__func__, '__qualname__', f'<No __qualname__: id:{id(f_cls.__func__)}>')
        if ref in _FUNCS:
            raise ConfigError(f'duplicate validator function "{ref}"; if this is intended, set `allow_reuse=True`')
        _FUNCS.add(ref)
    return f_cls