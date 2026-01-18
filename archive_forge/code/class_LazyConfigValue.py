from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
class LazyConfigValue(HasTraits):
    """Proxy object for exposing methods on configurable containers

    These methods allow appending/extending/updating
    to add to non-empty defaults instead of clobbering them.

    Exposes:

    - append, extend, insert on lists
    - update on dicts
    - update, add on sets
    """
    _value = None
    _extend: List[t.Any] = List()
    _prepend: List[t.Any] = List()
    _inserts: List[t.Any] = List()

    def append(self, obj: t.Any) -> None:
        """Append an item to a List"""
        self._extend.append(obj)

    def extend(self, other: t.Any) -> None:
        """Extend a list"""
        self._extend.extend(other)

    def prepend(self, other: t.Any) -> None:
        """like list.extend, but for the front"""
        self._prepend[:0] = other

    def merge_into(self, other: t.Any) -> t.Any:
        """
        Merge with another earlier LazyConfigValue or an earlier container.
        This is useful when having global system-wide configuration files.

        Self is expected to have higher precedence.

        Parameters
        ----------
        other : LazyConfigValue or container

        Returns
        -------
        LazyConfigValue
            if ``other`` is also lazy, a reified container otherwise.
        """
        if isinstance(other, LazyConfigValue):
            other._extend.extend(self._extend)
            self._extend = other._extend
            self._prepend.extend(other._prepend)
            other._inserts.extend(self._inserts)
            self._inserts = other._inserts
            if self._update:
                other.update(self._update)
                self._update = other._update
            return self
        else:
            return self.get_value(other)

    def insert(self, index: int, other: t.Any) -> None:
        if not isinstance(index, int):
            raise TypeError('An integer is required')
        self._inserts.append((index, other))
    _update = Any()

    def update(self, other: t.Any) -> None:
        """Update either a set or dict"""
        if self._update is None:
            if isinstance(other, dict):
                self._update = {}
            else:
                self._update = set()
        self._update.update(other)

    def add(self, obj: t.Any) -> None:
        """Add an item to a set"""
        self.update({obj})

    def get_value(self, initial: t.Any) -> t.Any:
        """construct the value from the initial one

        after applying any insert / extend / update changes
        """
        if self._value is not None:
            return self._value
        value = copy.deepcopy(initial)
        if isinstance(value, list):
            for idx, obj in self._inserts:
                value.insert(idx, obj)
            value[:0] = self._prepend
            value.extend(self._extend)
        elif isinstance(value, dict):
            if self._update:
                value.update(self._update)
        elif isinstance(value, set):
            if self._update:
                value.update(self._update)
        self._value = value
        return value

    def to_dict(self) -> dict[str, t.Any]:
        """return JSONable dict form of my data

        Currently update as dict or set, extend, prepend as lists, and inserts as list of tuples.
        """
        d = {}
        if self._update:
            d['update'] = self._update
        if self._extend:
            d['extend'] = self._extend
        if self._prepend:
            d['prepend'] = self._prepend
        elif self._inserts:
            d['inserts'] = self._inserts
        return d

    def __repr__(self) -> str:
        if self._value is not None:
            return f'<{self.__class__.__name__} value={self._value!r}>'
        else:
            return f'<{self.__class__.__name__} {self.to_dict()!r}>'