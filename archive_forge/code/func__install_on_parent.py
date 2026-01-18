import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os.path
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, Dict, Iterable, List, Optional, Union
from weakref import WeakValueDictionary
import torch
from torch.serialization import _get_restore_location, _maybe_decode_ascii
from ._directory_reader import DirectoryReader
from ._importlib import (
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
from .file_structure_representation import _create_directory_from_file_list, Directory
from .glob_group import GlobPattern
from .importer import Importer
def _install_on_parent(self, parent: str, name: str, module: types.ModuleType):
    if not parent:
        return
    parent_module = self.modules[parent]
    if parent_module.__loader__ is self:
        setattr(parent_module, name.rpartition('.')[2], module)