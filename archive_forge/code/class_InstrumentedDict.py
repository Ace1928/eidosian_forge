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
class InstrumentedDict(collections.UserDict):
    """
    Instrument an existing dictionary with additional
    functionality, but always reference and mutate
    the original dictionary.

    >>> orig = {'a': 1, 'b': 2}
    >>> inst = InstrumentedDict(orig)
    >>> inst['a']
    1
    >>> inst['c'] = 3
    >>> orig['c']
    3
    >>> inst.keys() == orig.keys()
    True
    """

    def __init__(self, data):
        super().__init__()
        self.data = data