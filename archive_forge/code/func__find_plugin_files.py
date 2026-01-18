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
def _find_plugin_files(self, src_dir: str) -> Iterable[tuple[str, str]]:
    """Get executable files from the plugins."""
    for plugin in self.plugins.file_tracers:
        for x_file in plugin.find_executable_files(src_dir):
            yield (x_file, plugin._coverage_plugin_name)