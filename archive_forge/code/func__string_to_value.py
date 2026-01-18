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
def _string_to_value(self, valuestr: str) -> Union[int, float, str, bool]:
    types = (int, float)
    for numtype in types:
        try:
            val = numtype(valuestr)
            if val != float(valuestr):
                continue
            return val
        except (ValueError, TypeError):
            continue
    vl = valuestr.lower()
    if vl == 'false':
        return False
    if vl == 'true':
        return True
    if not isinstance(valuestr, str):
        raise TypeError('Invalid value type: only int, long, float and str are allowed', valuestr)
    return valuestr