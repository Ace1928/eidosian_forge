from __future__ import annotations
from abc import ABCMeta, abstractmethod, abstractproperty
from inspect import isawaitable
from typing import (
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import FilterOrBool, Never, to_filter
from prompt_toolkit.keys import KEY_ALIASES, Keys
def _parse_key(key: Keys | str) -> str | Keys:
    """
    Replace key by alias and verify whether it's a valid one.
    """
    if isinstance(key, Keys):
        return key
    key = KEY_ALIASES.get(key, key)
    if key == 'space':
        key = ' '
    try:
        return Keys(key)
    except ValueError:
        pass
    if len(key) != 1:
        raise ValueError(f'Invalid key: {key}')
    return key