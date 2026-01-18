from __future__ import annotations
import importlib.machinery
import importlib.util
import inspect
import marshal
import os
import struct
import sys
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType
from typing import Any
from coverage import env
from coverage.exceptions import CoverageException, _ExceptionDuringRun, NoCode, NoSource
from coverage.files import canonical_filename, python_reported_file
from coverage.misc import isolate_module
from coverage.python import get_python_source
class DummyLoader:
    """A shim for the pep302 __loader__, emulating pkgutil.ImpLoader.

    Currently only implements the .fullname attribute
    """

    def __init__(self, fullname: str, *_args: Any) -> None:
        self.fullname = fullname