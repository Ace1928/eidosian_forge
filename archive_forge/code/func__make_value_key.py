from __future__ import annotations
import functools
import hashlib
import inspect
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Final
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import (
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import (
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
def _make_value_key(cache_type: CacheType, func: types.FunctionType, func_args: tuple[Any, ...], func_kwargs: dict[str, Any], hash_funcs: HashFuncsDict | None) -> str:
    """Create the key for a value within a cache.

    This key is generated from the function's arguments. All arguments
    will be hashed, except for those named with a leading "_".

    Raises
    ------
    StreamlitAPIException
        Raised (with a nicely-formatted explanation message) if we encounter
        an un-hashable arg.
    """
    arg_pairs: list[tuple[str | None, Any]] = []
    for arg_idx in range(len(func_args)):
        arg_name = _get_positional_arg_name(func, arg_idx)
        arg_pairs.append((arg_name, func_args[arg_idx]))
    for kw_name, kw_val in func_kwargs.items():
        arg_pairs.append((kw_name, kw_val))
    args_hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    for arg_name, arg_value in arg_pairs:
        if arg_name is not None and arg_name.startswith('_'):
            _LOGGER.debug('Not hashing %s because it starts with _', arg_name)
            continue
        try:
            update_hash(arg_name, hasher=args_hasher, cache_type=cache_type, hash_source=func)
            update_hash(arg_value, hasher=args_hasher, cache_type=cache_type, hash_funcs=hash_funcs, hash_source=func)
        except UnhashableTypeError as exc:
            raise UnhashableParamError(cache_type, func, arg_name, arg_value, exc)
    value_key = args_hasher.hexdigest()
    _LOGGER.debug('Cache key: %s', value_key)
    return value_key