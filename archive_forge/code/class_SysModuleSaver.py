from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
class SysModuleSaver:
    """Saves the contents of sys.modules, and removes new modules later."""

    def __init__(self) -> None:
        self.old_modules = set(sys.modules)

    def restore(self) -> None:
        """Remove any modules imported since this object started."""
        new_modules = set(sys.modules) - self.old_modules
        for m in new_modules:
            del sys.modules[m]