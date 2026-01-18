from __future__ import annotations
import collections.abc
import copy
import functools
import itertools
import operator
import random
import re
from collections.abc import Container, Iterable, Mapping
from typing import Any, Callable, Union
import jaraco.text
class ItemsAsAttributes:
    """
    Mix-in class to enable a mapping object to provide items as
    attributes.

    >>> C = type('C', (dict, ItemsAsAttributes), dict())
    >>> i = C()
    >>> i['foo'] = 'bar'
    >>> i.foo
    'bar'

    Natural attribute access takes precedence

    >>> i.foo = 'henry'
    >>> i.foo
    'henry'

    But as you might expect, the mapping functionality is preserved.

    >>> i['foo']
    'bar'

    A normal attribute error should be raised if an attribute is
    requested that doesn't exist.

    >>> i.missing
    Traceback (most recent call last):
    ...
    AttributeError: 'C' object has no attribute 'missing'

    It also works on dicts that customize __getitem__

    >>> missing_func = lambda self, key: 'missing item'
    >>> C = type(
    ...     'C',
    ...     (dict, ItemsAsAttributes),
    ...     dict(__missing__ = missing_func),
    ... )
    >>> i = C()
    >>> i.missing
    'missing item'
    >>> i.foo
    'missing item'
    """

    def __getattr__(self, key):
        try:
            return getattr(super(), key)
        except AttributeError as e:
            noval = object()

            def _safe_getitem(cont, key, missing_result):
                try:
                    return cont[key]
                except KeyError:
                    return missing_result
            result = _safe_getitem(self, key, noval)
            if result is not noval:
                return result
            message, = e.args
            message = message.replace('super', self.__class__.__name__, 1)
            e.args = (message,)
            raise