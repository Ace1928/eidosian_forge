import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
@dataclass(**_LOADED_INODE_DATACLASS_ARGS)
class _LoadedInode:
    dev: int
    inode: int