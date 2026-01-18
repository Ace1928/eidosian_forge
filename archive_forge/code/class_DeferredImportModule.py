from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class DeferredImportModule(object):
    """Mock module object to support the deferred import of a module.

    This object is returned by :py:func:`attempt_import()` in lieu of
    the module when :py:func:`attempt_import()` is called with
    ``defer_check=True``.  Any attempts to access attributes on this
    object will trigger the actual module import and return either the
    appropriate module attribute or else if the module import fails,
    raise a :py:class:`.DeferredImportError` exception.

    """

    def __init__(self, indicator, deferred_submodules, submodule_name):
        self._indicator_flag = indicator
        self._submodule_name = submodule_name
        self.__file__ = None
        self.__spec__ = None
        if not deferred_submodules:
            return
        if submodule_name is None:
            submodule_name = ''
        for name in deferred_submodules:
            if not name.startswith(submodule_name + '.'):
                continue
            _local_name = name[1 + len(submodule_name):]
            if '.' in _local_name:
                continue
            setattr(self, _local_name, DeferredImportModule(indicator, deferred_submodules, submodule_name + '.' + _local_name))

    def __getattr__(self, attr):
        self._indicator_flag.resolve()
        _mod = self._indicator_flag._module
        if self._submodule_name:
            for _sub in self._submodule_name[1:].split('.'):
                _mod = getattr(_mod, _sub)
        return getattr(_mod, attr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for k, v in state.items():
            super().__setattr__(k, v)

    def mro(self):
        return [DeferredImportModule, object]