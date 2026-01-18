import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
def _is_torchpackage_dummy(self, module):
    """Returns true iff this module is an empty PackageNode in a torch.package.

        If you intern `a.b` but never use `a` in your code, then `a` will be an
        empty module with no source. This can break cases where we are trying to
        re-package an object after adding a real dependency on `a`, since
        OrderedImportere will resolve `a` to the dummy package and stop there.

        See: https://github.com/pytorch/pytorch/pull/71520#issuecomment-1029603769
        """
    if not getattr(module, '__torch_package__', False):
        return False
    if not hasattr(module, '__path__'):
        return False
    if not hasattr(module, '__file__'):
        return True
    return module.__file__ is None