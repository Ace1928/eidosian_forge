import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method
class LazyImportTester(LazyDependencyManager):
    """A lazy dependency tester for importable Python modules.  Any required objects will only be
    imported at the point that this object is tested for its Boolean value."""
    __slots__ = ('_modules',)

    def __init__(self, name_map_or_modules: Union[str, Dict[str, Iterable[str]], Iterable[str]], *, name: Optional[str]=None, callback: Optional[Callable[[bool], None]]=None, install: Optional[str]=None, msg: Optional[str]=None):
        """
        Args:
            name_map_or_modules: if a name map, then a dictionary where the keys are modules or
                packages, and the values are iterables of names to try and import from that
                module.  It should be valid to write ``from <module> import <name1>, <name2>, ...``.
                If simply a string or iterable of strings, then it should be valid to write
                ``import <module>`` for each of them.

        Raises:
            ValueError: if no modules are given.
        """
        if isinstance(name_map_or_modules, dict):
            self._modules = {module: tuple(names) for module, names in name_map_or_modules.items()}
        elif isinstance(name_map_or_modules, str):
            self._modules = {name_map_or_modules: ()}
        else:
            self._modules = {module: () for module in name_map_or_modules}
        if not self._modules:
            raise ValueError('no modules supplied')
        if name is not None:
            pass
        elif len(self._modules) == 1:
            name, = self._modules.keys()
        else:
            all_names = tuple(self._modules.keys())
            name = f'{', '.join(all_names[:-1])} and {all_names[-1]}'
        super().__init__(name=name, callback=callback, install=install, msg=msg)

    def _is_available(self):
        failed_modules = {}
        failed_names = collections.defaultdict(list)
        for module, names in self._modules.items():
            try:
                imported = importlib.import_module(module)
            except ModuleNotFoundError as exc:
                failed_parts = exc.name.split('.')
                target_parts = module.split('.')
                if failed_parts == target_parts[:len(failed_parts)]:
                    return False
                failed_modules[module] = exc
                continue
            except ImportError as exc:
                failed_modules[module] = exc
                continue
            for name in names:
                try:
                    _ = getattr(imported, name)
                except AttributeError:
                    failed_names[module].append(name)
        if failed_modules or failed_names:
            package_description = f"'{self._name}'" if self._name else 'optional packages'
            message = f'While trying to import {package_description}, some components were located but raised other errors during import. You might have an incompatible version installed. Qiskit will continue as if the optional is not available.'
            for module, exc in failed_modules.items():
                message += ''.join(f"\n - module '{module}' failed to import with: {exc!r}")
            for module, names in failed_names.items():
                attributes = f"attribute '{names[0]}'" if len(names) == 1 else f'attributes {names}'
                message += ''.join(f"\n - '{module}' imported, but {attributes} couldn't be found")
            warnings.warn(message, category=OptionalDependencyImportWarning)
            return False
        return True