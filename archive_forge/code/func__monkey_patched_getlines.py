import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List
import triton
def _monkey_patched_getlines(filename, module_globals=None):
    if filename in _FILENAME_TO_SRC:
        return _FILENAME_TO_SRC[filename]
    else:
        return _getlines_orig(filename, module_globals)