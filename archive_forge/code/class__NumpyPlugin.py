from __future__ import annotations
from collections.abc import Iterable
from typing import Final, TYPE_CHECKING, Callable
import numpy as np
class _NumpyPlugin(Plugin):
    """A mypy plugin for handling versus numpy-specific typing tasks."""

    def get_type_analyze_hook(self, fullname: str) -> None | _HookFunc:
        """Set the precision of platform-specific `numpy.number`
            subclasses.

            For example: `numpy.int_`, `numpy.longlong` and `numpy.longdouble`.
            """
        if fullname in _PRECISION_DICT:
            return _hook
        return None

    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        """Handle all import-based overrides.

            * Import platform-specific extended-precision `numpy.number`
              subclasses (*e.g.* `numpy.float96`, `numpy.float128` and
              `numpy.complex256`).
            * Import the appropriate `ctypes` equivalent to `numpy.intp`.

            """
        ret = [(PRI_MED, file.fullname, -1)]
        if file.fullname == 'numpy':
            _override_imports(file, 'numpy._typing._extended_precision', imports=[(v, v) for v in _EXTENDED_PRECISION_LIST])
        elif file.fullname == 'numpy.ctypeslib':
            _override_imports(file, 'ctypes', imports=[(_C_INTP, '_c_intp')])
        return ret