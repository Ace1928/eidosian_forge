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
class Everything:
    """
    A collection "containing" every possible thing.

    >>> 'foo' in Everything()
    True

    >>> import random
    >>> random.randint(1, 999) in Everything()
    True

    >>> random.choice([None, 'foo', 42, ('a', 'b', 'c')]) in Everything()
    True
    """

    def __contains__(self, other):
        return True