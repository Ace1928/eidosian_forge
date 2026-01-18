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
def add_coverage_paths(paths: set[str]) -> None:
    """Add paths where coverage.py code can be found to the set `paths`."""
    cover_path = canonical_path(__file__, directory=True)
    paths.add(cover_path)
    if env.TESTING:
        paths.add(os.path.join(cover_path, 'tests'))