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
class WeightedLookup(RangeMap):
    """
    Given parameters suitable for a dict representing keys
    and a weighted proportion, return a RangeMap representing
    spans of values proportial to the weights:

    >>> even = WeightedLookup(a=1, b=1)

    [0, 1) -> a
    [1, 2) -> b

    >>> lk = WeightedLookup(a=1, b=2)

    [0, 1) -> a
    [1, 3) -> b

    >>> lk[.5]
    'a'
    >>> lk[1.5]
    'b'

    Adds ``.random()`` to select a random weighted value:

    >>> lk.random() in ['a', 'b']
    True

    >>> choices = [lk.random() for x in range(1000)]

    Statistically speaking, choices should be .5 a:b
    >>> ratio = choices.count('a') / choices.count('b')
    >>> .4 < ratio < .6
    True
    """

    def __init__(self, *args, **kwargs):
        raw = dict(*args, **kwargs)
        indexes = map(Accumulator(), raw.values())
        super().__init__(zip(indexes, raw.keys()), key_match_comparator=operator.lt)

    def random(self):
        lower, upper = self.bounds()
        selector = random.random() * upper
        return self[selector]