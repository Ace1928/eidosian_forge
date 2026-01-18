import importlib
from abc import ABC, abstractmethod
from pickle import (  # type: ignore[attr-defined]  # type: ignore[attr-defined]
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from ._mangling import demangle, get_mangle_prefix, is_mangled
class OrderedImporter(Importer):
    """A compound importer that takes a list of importers and tries them one at a time.

    The first importer in the list that returns a result "wins".
    """

    def __init__(self, *args):
        self._importers: List[Importer] = list(args)

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

    def import_module(self, module_name: str) -> ModuleType:
        last_err = None
        for importer in self._importers:
            if not isinstance(importer, Importer):
                raise TypeError(f'{importer} is not a Importer. All importers in OrderedImporter must inherit from Importer.')
            try:
                module = importer.import_module(module_name)
                if self._is_torchpackage_dummy(module):
                    continue
                return module
            except ModuleNotFoundError as err:
                last_err = err
        if last_err is not None:
            raise last_err
        else:
            raise ModuleNotFoundError(module_name)

    def whichmodule(self, obj: Any, name: str) -> str:
        for importer in self._importers:
            module_name = importer.whichmodule(obj, name)
            if module_name != '__main__':
                return module_name
        return '__main__'