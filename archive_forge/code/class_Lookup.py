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
class Lookup:
    """
    A micro-optimized class for searching a (fast) path for metadata.
    """

    def __init__(self, path: FastPath):
        """
        Calculate all of the children representing metadata.

        From the children in the path, calculate early all of the
        children that appear to represent metadata (infos) or legacy
        metadata (eggs).
        """
        base = os.path.basename(path.root).lower()
        base_is_egg = base.endswith('.egg')
        self.infos = FreezableDefaultDict(list)
        self.eggs = FreezableDefaultDict(list)
        for child in path.children():
            low = child.lower()
            if low.endswith(('.dist-info', '.egg-info')):
                name = low.rpartition('.')[0].partition('-')[0]
                normalized = Prepared.normalize(name)
                self.infos[normalized].append(path.joinpath(child))
            elif base_is_egg and low == 'egg-info':
                name = base.rpartition('.')[0].partition('-')[0]
                legacy_normalized = Prepared.legacy_normalize(name)
                self.eggs[legacy_normalized].append(path.joinpath(child))
        self.infos.freeze()
        self.eggs.freeze()

    def search(self, prepared: Prepared):
        """
        Yield all infos and eggs matching the Prepared query.
        """
        infos = self.infos[prepared.normalized] if prepared else itertools.chain.from_iterable(self.infos.values())
        eggs = self.eggs[prepared.legacy_normalized] if prepared else itertools.chain.from_iterable(self.eggs.values())
        return itertools.chain(infos, eggs)