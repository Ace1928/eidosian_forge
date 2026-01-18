from __future__ import annotations
import importlib.util
import inspect
import itertools
import os
import platform
import re
import sys
import sysconfig
import traceback
from types import FrameType, ModuleType
from typing import (
from coverage import env
from coverage.disposition import FileDisposition, disposition_init
from coverage.exceptions import CoverageException, PluginError
from coverage.files import TreeMatcher, GlobMatcher, ModuleMatcher
from coverage.files import prep_patterns, find_python_files, canonical_filename
from coverage.misc import sys_modules_saved
from coverage.python import source_for_file, source_for_morf
from coverage.types import TFileDisposition, TMorf, TWarnFn, TDebugCtl
def _find_executable_files(self, src_dir: str) -> Iterable[tuple[str, str | None]]:
    """Find executable files in `src_dir`.

        Search for files in `src_dir` that can be executed because they
        are probably importable. Don't include ones that have been omitted
        by the configuration.

        Yield the file path, and the plugin name that handles the file.

        """
    py_files = ((py_file, None) for py_file in find_python_files(src_dir, self.include_namespace_packages))
    plugin_files = self._find_plugin_files(src_dir)
    for file_path, plugin_name in itertools.chain(py_files, plugin_files):
        file_path = canonical_filename(file_path)
        if self.omit_match and self.omit_match.match(file_path):
            continue
        yield (file_path, plugin_name)