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
def _do_find_and_load(self, name):
    path = None
    parent = name.rpartition('.')[0]
    module_name_no_parent = name.rpartition('.')[-1]
    if parent:
        if parent not in self.modules:
            self._gcd_import(parent)
        if name in self.modules:
            return self.modules[name]
        parent_module = self.modules[parent]
        try:
            path = parent_module.__path__
        except AttributeError:
            if isinstance(parent_module.__loader__, importlib.machinery.ExtensionFileLoader):
                if name not in self.extern_modules:
                    msg = (_ERR_MSG + '; {!r} is a c extension module which was not externed. C extension modules                             need to be externed by the PackageExporter in order to be used as we do not support interning them.}.').format(name, name)
                    raise ModuleNotFoundError(msg, name=name) from None
                if not isinstance(parent_module.__dict__.get(module_name_no_parent), types.ModuleType):
                    msg = (_ERR_MSG + '; {!r} is a c extension package which does not contain {!r}.').format(name, parent, name)
                    raise ModuleNotFoundError(msg, name=name) from None
            else:
                msg = (_ERR_MSG + '; {!r} is not a package').format(name, parent)
                raise ModuleNotFoundError(msg, name=name) from None
    module = self._load_module(name, parent)
    self._install_on_parent(parent, name, module)
    return module