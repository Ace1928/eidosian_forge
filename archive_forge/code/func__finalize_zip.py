import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import (
import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle
from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer
from _mock import MockedObject
def _finalize_zip(self):
    """Called at the very end of packaging to leave the zipfile in a closed but valid state."""
    del self.zip_file
    if self.buffer:
        self.buffer.flush()