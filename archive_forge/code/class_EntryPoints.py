from __future__ import annotations
import os
import re
import abc
import csv
import sys
import json
import zipp
import email
import types
import inspect
import pathlib
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections
from . import _adapters, _meta, _py39compat
from ._collections import FreezableDefaultDict, Pair
from ._compat import (
from ._functools import method_cache, pass_none
from ._itertools import always_iterable, unique_everseen
from ._meta import PackageMetadata, SimplePath
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, Iterable, List, Mapping, Match, Optional, Set, cast
class EntryPoints(tuple):
    """
    An immutable collection of selectable EntryPoint objects.
    """
    __slots__ = ()

    def __getitem__(self, name: str) -> EntryPoint:
        """
        Get the EntryPoint in self matching name.
        """
        try:
            return next(iter(self.select(name=name)))
        except StopIteration:
            raise KeyError(name)

    def __repr__(self):
        """
        Repr with classname and tuple constructor to
        signal that we deviate from regular tuple behavior.
        """
        return '%s(%r)' % (self.__class__.__name__, tuple(self))

    def select(self, **params) -> EntryPoints:
        """
        Select entry points from self that match the
        given parameters (typically group and/or name).
        """
        return EntryPoints((ep for ep in self if _py39compat.ep_matches(ep, **params)))

    @property
    def names(self) -> Set[str]:
        """
        Return the set of all names of all entry points.
        """
        return {ep.name for ep in self}

    @property
    def groups(self) -> Set[str]:
        """
        Return the set of all groups of all entry points.
        """
        return {ep.group for ep in self}

    @classmethod
    def _from_text_for(cls, text, dist):
        return cls((ep._for(dist) for ep in cls._from_text(text)))

    @staticmethod
    def _from_text(text):
        return (EntryPoint(name=item.value.name, value=item.value.value, group=item.name) for item in Sectioned.section_pairs(text or ''))