from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import importlib
import logging
import os
import sys
import threading
from googlecloudsdk.core.util import encoding
class ConfigHandle(object):
    """A set of configuration for a single library module or package.

  Public attributes of instances of this class are configuration values.
  Attributes are dynamically computed (in `__getattr__()`) and cached as regular
  instance attributes.
  """
    _initialized = False

    def __init__(self, prefix, registry):
        """Constructor.

    Args:
      prefix: A shared prefix for the configuration names being registered. It
          must end in `_`. This requirement is enforced by
          `LibConfigRegistry`.
      registry: A `LibConfigRegistry` instance.
    """
        assert prefix.endswith('_')
        self._prefix = prefix
        self._defaults = {}
        self._overrides = {}
        self._registry = registry
        self._lock = threading.RLock()

    def _update_defaults(self, mapping):
        """Updates the default mappings.

    Args:
      mapping: A dict mapping suffix strings to default values.
    """
        self._lock.acquire()
        try:
            for key, value in mapping.items():
                if key.startswith('__') and key.endswith('__'):
                    continue
                self._defaults[key] = value
            if self._initialized:
                self._update_configs()
        finally:
            self._lock.release()

    def _update_configs(self):
        """Updates the configuration values.

    This clears the cached values, initializes the registry, and loads
    the configuration values from the config module.
    """
        self._lock.acquire()
        try:
            if self._initialized:
                self._clear_cache()
            self._registry.initialize()
            for key, value in self._registry._pairs(self._prefix):
                if key not in self._defaults:
                    logging.warn('Configuration "%s" not recognized', self._prefix + key)
                else:
                    self._overrides[key] = value
            self._initialized = True
        finally:
            self._lock.release()

    def _clear_cache(self):
        """Clears the cached values."""
        self._lock.acquire()
        try:
            self._initialized = False
            for key in self._defaults:
                self._overrides.pop(key, None)
                try:
                    delattr(self, key)
                except AttributeError:
                    pass
        finally:
            self._lock.release()

    def _dump(self):
        """Prints information about this set of registrations to stdout."""
        self._lock.acquire()
        try:
            print('Prefix %s:' % self._prefix)
            if self._overrides:
                print('  Overrides:')
                for key in sorted(self._overrides):
                    print('    %s = %r' % (key, self._overrides[key]))
            else:
                print('  No overrides')
            if self._defaults:
                print('  Defaults:')
                for key in sorted(self._defaults):
                    print('    %s = %r' % (key, self._defaults[key]))
            else:
                print('  No defaults')
            print('-' * 40)
        finally:
            self._lock.release()

    def __getattr__(self, suffix):
        """Dynamic attribute access.

    Args:
      suffix: The attribute name.

    Returns:
      A configuration values.

    Raises:
      AttributeError: If the suffix is not a registered suffix.

    The first time an attribute is referenced, this method is invoked. The value
    returned is taken either from the config module or from the registered
    default.
    """
        self._lock.acquire()
        try:
            if not self._initialized:
                self._update_configs()
            if suffix in self._overrides:
                value = self._overrides[suffix]
            elif suffix in self._defaults:
                value = self._defaults[suffix]
            else:
                raise AttributeError(suffix)
            setattr(self, suffix, value)
            return value
        finally:
            self._lock.release()