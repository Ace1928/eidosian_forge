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
class IdentityOverrideMap(dict):
    """
    A dictionary that by default maps each key to itself, but otherwise
    acts like a normal dictionary.

    >>> d = IdentityOverrideMap()
    >>> d[42]
    42
    >>> d['speed'] = 'speedo'
    >>> print(d['speed'])
    speedo
    """

    def __missing__(self, key):
        return key