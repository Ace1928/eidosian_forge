import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def _call_config(self, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call the configuration at the given method which must take a section name
        as first argument."""
    return getattr(self._config, method)(self._section_name, *args, **kwargs)