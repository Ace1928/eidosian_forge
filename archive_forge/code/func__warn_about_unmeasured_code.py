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
def _warn_about_unmeasured_code(self, pkg: str) -> None:
    """Warn about a package or module that we never traced.

        `pkg` is a string, the name of the package or module.

        """
    mod = sys.modules.get(pkg)
    if mod is None:
        self.warn(f'Module {pkg} was never imported.', slug='module-not-imported')
        return
    if module_is_namespace(mod):
        return
    if not module_has_file(mod):
        self.warn(f'Module {pkg} has no Python source.', slug='module-not-python')
        return
    msg = f'Module {pkg} was previously imported, but not measured'
    self.warn(msg, slug='module-not-measured')