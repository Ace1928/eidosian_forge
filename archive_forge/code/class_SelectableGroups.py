import os
import re
import abc
import csv
import sys
import email
import pathlib
import zipfile
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections
from . import _adapters, _meta
from ._collections import FreezableDefaultDict, Pair
from ._functools import method_cache, pass_none
from ._itertools import always_iterable, unique_everseen
from ._meta import PackageMetadata, SimplePath
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import List, Mapping, Optional, Union
class SelectableGroups(Deprecated, dict):
    """
    A backward- and forward-compatible result from
    entry_points that fully implements the dict interface.
    """

    @classmethod
    def load(cls, eps):
        by_group = operator.attrgetter('group')
        ordered = sorted(eps, key=by_group)
        grouped = itertools.groupby(ordered, by_group)
        return cls(((group, EntryPoints(eps)) for group, eps in grouped))

    @property
    def _all(self):
        """
        Reconstruct a list of all entrypoints from the groups.
        """
        groups = super(Deprecated, self).values()
        return EntryPoints(itertools.chain.from_iterable(groups))

    @property
    def groups(self):
        return self._all.groups

    @property
    def names(self):
        """
        for coverage:
        >>> SelectableGroups().names
        set()
        """
        return self._all.names

    def select(self, **params):
        if not params:
            return self
        return self._all.select(**params)