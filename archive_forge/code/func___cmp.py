from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def __cmp(self, other: 'Version', comparator: T.Callable[[T.Any, T.Any], bool]) -> bool:
    for ours, theirs in zip(self._v, other._v):
        ours_is_int = isinstance(ours, int)
        theirs_is_int = isinstance(theirs, int)
        if ours_is_int != theirs_is_int:
            return comparator(ours_is_int, theirs_is_int)
        if ours != theirs:
            return comparator(ours, theirs)
    return comparator(len(self._v), len(other._v))