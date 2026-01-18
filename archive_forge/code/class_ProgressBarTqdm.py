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
class ProgressBarTqdm(tqdm):

    def __init__(self, *args: T.Any, bar_type: T.Optional[str]=None, **kwargs: T.Any) -> None:
        if bar_type == 'download':
            kwargs.update({'unit': 'B', 'unit_scale': True, 'unit_divisor': 1024, 'leave': True, 'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt} eta {remaining}'})
        else:
            kwargs.update({'leave': False, 'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} eta {remaining}'})
        super().__init__(*args, **kwargs)