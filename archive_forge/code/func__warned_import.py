from typing import Any, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union
import warnings
from gitdb.util import to_hex_sha
from git.exc import (
from git.types import PathLike
def _warned_import(message: str, fullname: str) -> 'ModuleType':
    import importlib
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    return importlib.import_module(fullname)