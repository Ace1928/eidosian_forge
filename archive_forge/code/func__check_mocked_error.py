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
def _check_mocked_error(module: Optional[str], field: Optional[str]):
    """
            checks if an object (field) comes from a mocked module and then adds
            the pair to mocked_modules which contains mocked modules paired with their
            list of mocked objects present in the pickle.

            We also hold the invariant that the first user defined rule that applies
            to the module is the one we use.
            """
    assert isinstance(module, str)
    assert isinstance(field, str)
    if self._can_implicitly_extern(module):
        return
    for pattern, pattern_info in self.patterns.items():
        if pattern.matches(module):
            if pattern_info.action == _ModuleProviderAction.MOCK:
                mocked_modules[module].append(field)
            return