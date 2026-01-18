from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
class LibConfigRegistry(object):
    """A registry containing library configuration values."""

    def __init__(self, modname):
        """Constructor.

    Args:
      modname: The module name to be imported.

    Note: the actual import of this module is deferred until the first
    time a configuration value is requested through attribute access
    on a ConfigHandle instance.
    """
        self._modname = modname
        self._registrations = {}
        self._module = None
        self._lock = threading.RLock()

    def register(self, prefix, mapping):
        """Registers a set of configuration names.

    Args:
      prefix: A shared prefix for the configuration names being registered.
          If the prefix doesn't end in `_`, that character is appended.
      mapping: A dict that maps suffix strings to default values.

    Returns:
      A `ConfigHandle` instance.

    You can re-register the same prefix: the mappings are merged, and for
    duplicate suffixes, the most recent registration is used.
    """
        if not prefix.endswith('_'):
            prefix += '_'
        self._lock.acquire()
        try:
            handle = self._registrations.get(prefix)
            if handle is None:
                handle = ConfigHandle(prefix, self)
                self._registrations[prefix] = handle
        finally:
            self._lock.release()
        handle._update_defaults(mapping)
        return handle

    def initialize(self, import_func=importlib.import_module):
        """Tries to import the configuration module if it is not already imported.

    This function always sets `self._module` to a value that is not `None`;
    either the imported module (if it was imported successfully) or a placeholder
    `object()` instance (if an `ImportError` was raised) is used. Other
    exceptions are not caught.

    When a placeholder instance is used, the instance is also put in `sys.modules`.
    This usage allows us to detect when `sys.modules` was changed (as
    `dev_appserver.py` does when it notices source code changes) and retries the
    `import_module` in that case, while skipping it (for speed) if nothing has
    changed.

    Args:
      import_func: Used for dependency injection.
    """
        self._lock.acquire()
        try:
            if self._module is not None and self._module is sys.modules.get(self._modname):
                return
            try:
                import_func(self._modname)
            except ImportError as err:
                if str(err) not in ['No module named {}'.format(self._modname), 'import of {} halted; None in sys.modules'.format(self._modname)]:
                    raise
                self._module = object()
                sys.modules[self._modname] = self._module
            else:
                self._module = sys.modules[self._modname]
        finally:
            self._lock.release()

    def reset(self):
        """Drops the imported configuration module.

    If the configuration module has not been imported, no operation occurs, and
    the next operation takes place.
    """
        self._lock.acquire()
        try:
            if self._module is None:
                return
            self._module = None
            handles = list(self._registrations.values())
        finally:
            self._lock.release()
        for handle in handles:
            handle._clear_cache()

    def _pairs(self, prefix):
        """Generates `(key, value)` pairs from the config module matching prefix.

    Args:
      prefix: A prefix string ending in `_`, for example: `mylib_`.

    Yields:
      `(key, value)` pairs, where `key` is the configuration name with the
      prefix removed, and `value` is the corresponding value.
    """
        self._lock.acquire()
        try:
            mapping = getattr(self._module, '__dict__', None)
            if not mapping:
                return
            items = list(mapping.items())
        finally:
            self._lock.release()
        nskip = len(prefix)
        for key, value in items:
            if key.startswith(prefix):
                yield (key[nskip:], value)

    def _dump(self):
        """Prints information about all registrations to stdout."""
        self.initialize()
        handles = []
        self._lock.acquire()
        try:
            if not hasattr(self._module, '__dict__'):
                print('Module %s.py does not exist.' % self._modname)
            elif not self._registrations:
                print('No registrations for %s.py.' % self._modname)
            else:
                print('Registrations in %s.py:' % self._modname)
                print('-' * 40)
                handles = list(self._registrations.items())
        finally:
            self._lock.release()
        for _, handle in sorted(handles):
            handle._dump()