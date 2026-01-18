from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def _enable_vendoring() -> None:
    """Enable support for loading Python packages vendored by ansible-core."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        load_module(os.path.join(ANSIBLE_LIB_ROOT, '_vendor', '__init__.py'), 'ansible_vendor')