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
def _find_libraries_pyodide(self):
    """Pyodide specific implementation for finding loaded libraries.

        Adapted from suggestion in https://github.com/joblib/threadpoolctl/pull/169#issuecomment-1946696449.

        One day, we may have a simpler solution. libc dl_iterate_phdr needs to
        be implemented in Emscripten and exposed in Pyodide, see
        https://github.com/emscripten-core/emscripten/issues/21354 for more
        details.
        """
    try:
        from pyodide_js._module import LDSO
    except ImportError:
        warnings.warn('Unable to import LDSO from pyodide_js._module. This should never happen.')
        return
    for filepath in LDSO.loadedLibsByName.as_object_map():
        if os.path.exists(filepath):
            self._make_controller_from_path(filepath)