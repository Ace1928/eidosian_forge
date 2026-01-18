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
def _intern_module(self, module_name: str, dependencies: bool):
    """Adds the module to the dependency graph as an interned module,
        along with any metadata needed to write it out to the zipfile at serialization time.
        """
    module_obj = self._import_module(module_name)
    module_name = demangle(module_name)
    is_package = hasattr(module_obj, '__path__')
    source = self._get_source_of_module(module_obj)
    if source is None:
        filename = getattr(module_obj, '__file__', None)
        error_context = None
        if filename is None:
            packaging_error = PackagingErrorReason.NO_DUNDER_FILE
        elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
        else:
            packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
            error_context = f'filename: {filename}'
        self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.INTERN, is_package=is_package, error=packaging_error, error_context=error_context, provided=True)
        return
    self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.INTERN, is_package=is_package, source=source, provided=True)
    if dependencies:
        deps = self._get_dependencies(source, module_name, is_package)
        for dep in deps:
            self.dependency_graph.add_edge(module_name, dep)
            self.add_dependency(dep)