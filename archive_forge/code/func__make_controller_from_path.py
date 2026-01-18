import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def _make_controller_from_path(self, filepath):
    """Store a library controller if it is supported and selected"""
    filepath = _realpath(filepath)
    filename = os.path.basename(filepath).lower()
    for controller_class in _ALL_CONTROLLERS:
        prefix = self._check_prefix(filename, controller_class.filename_prefixes)
        if prefix is None:
            continue
        if prefix == 'libblas':
            if filename.endswith('.dll'):
                libblas = ctypes.CDLL(filepath, _RTLD_NOLOAD)
                if not any((hasattr(libblas, func) for func in controller_class.check_symbols)):
                    continue
            else:
                continue
        lib_controller = controller_class(filepath=filepath, prefix=prefix, parent=self)
        if filepath in (lib.filepath for lib in self.lib_controllers):
            continue
        if not hasattr(controller_class, 'check_symbols') or any((hasattr(lib_controller.dynlib, func) for func in controller_class.check_symbols)):
            self.lib_controllers.append(lib_controller)