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
class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):
        return f'<{self.__class__.__name__}.{self.name}>'
    IS_EXTENSION_MODULE = 'Module is a C extension module. torch.package supports Python modules only.'
    NO_DUNDER_FILE = 'Module had no __file__ defined.'
    SOURCE_FILE_NOT_FOUND = 'Module had a __file__, but we could not find it in your filesystem.'
    DEPENDENCY_RESOLUTION_FAILED = 'Dependency resolution failed.'
    NO_ACTION = 'Module did not match against any action pattern. Extern, mock, or intern it.'
    DENIED = 'Module was denied by a pattern.'
    MOCKED_BUT_STILL_USED = 'Module was mocked out, but is still being used in the package. Please intern or extern the mocked modules if objects are supposed to be in the package.'