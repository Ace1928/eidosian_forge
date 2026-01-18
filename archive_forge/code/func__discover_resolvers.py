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
@staticmethod
def _discover_resolvers():
    """Search the meta_path for resolvers (MetadataPathFinders)."""
    declared = (getattr(finder, 'find_distributions', None) for finder in sys.meta_path)
    return filter(None, declared)