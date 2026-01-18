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
class PackagePath(pathlib.PurePosixPath):
    """A reference to a path in a package"""
    hash: Optional[FileHash]
    size: int
    dist: Distribution

    def read_text(self, encoding: str='utf-8') -> str:
        return self.locate().read_text(encoding=encoding)

    def read_binary(self) -> bytes:
        return self.locate().read_bytes()

    def locate(self) -> SimplePath:
        """Return a path-like object for this path"""
        return self.dist.locate_file(self)