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
def _persistent_id(self, obj):
    if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
        storage: Storage
        if isinstance(obj, torch.storage.TypedStorage):
            untyped_storage = obj._untyped_storage
            storage_type_str = obj.pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage = cast(Storage, untyped_storage)
            storage_numel = obj.size()
        elif isinstance(obj, torch.UntypedStorage):
            untyped_storage = obj
            storage = cast(Storage, untyped_storage)
            storage_type = normalize_storage_type(type(storage))
            storage_numel = storage.nbytes()
        else:
            raise RuntimeError(f'storage type not recognized: {type(obj)}')
        location = location_tag(storage)
        storage_present = self.storage_context.has_storage(storage)
        storage_id = self.storage_context.get_or_add_storage(storage)
        if not storage_present:
            if storage.device.type != 'cpu':
                storage = storage.cpu()
            num_bytes = storage.nbytes()
            self.zip_file.write_record(f'.data/{storage_id}.storage', storage.data_ptr(), num_bytes)
        return ('storage', storage_type, storage_id, location, storage_numel)
    if hasattr(obj, '__reduce_package__'):
        if _gate_torchscript_serialization and isinstance(obj, torch.jit.RecursiveScriptModule):
            raise Exception('Serializing ScriptModules directly into a package is a beta feature. To use, set global `torch.package.package_exporter._gate_torchscript_serialization` to `False`.')
        if self.serialized_reduces.get(id(obj)) is None:
            self.serialized_reduces[id(obj)] = ('reduce_package', id(obj), *obj.__reduce_package__(self))
        return self.serialized_reduces[id(obj)]
    return None