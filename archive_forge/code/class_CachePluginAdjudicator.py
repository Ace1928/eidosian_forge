from __future__ import (absolute_import, division, print_function)
import copy
import errno
import os
import tempfile
import time
from abc import abstractmethod
from collections.abc import MutableMapping
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins import AnsiblePlugin
from ansible.plugins.loader import cache_loader
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
class CachePluginAdjudicator(MutableMapping):
    """
    Intermediary between a cache dictionary and a CacheModule
    """

    def __init__(self, plugin_name='memory', **kwargs):
        self._cache = {}
        self._retrieved = {}
        self._plugin = cache_loader.get(plugin_name, **kwargs)
        if not self._plugin:
            raise AnsibleError('Unable to load the cache plugin (%s).' % plugin_name)
        self._plugin_name = plugin_name

    def update_cache_if_changed(self):
        if self._retrieved != self._cache:
            self.set_cache()

    def set_cache(self):
        for top_level_cache_key in self._cache.keys():
            self._plugin.set(top_level_cache_key, self._cache[top_level_cache_key])
        self._retrieved = copy.deepcopy(self._cache)

    def load_whole_cache(self):
        for key in self._plugin.keys():
            self._cache[key] = self._plugin.get(key)

    def __repr__(self):
        return to_text(self._cache)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def _do_load_key(self, key):
        load = False
        if all([key not in self._cache, key not in self._retrieved, self._plugin_name != 'memory', self._plugin.contains(key)]):
            load = True
        return load

    def __getitem__(self, key):
        if self._do_load_key(key):
            try:
                self._cache[key] = self._plugin.get(key)
            except KeyError:
                pass
            else:
                self._retrieved[key] = self._cache[key]
        return self._cache[key]

    def get(self, key, default=None):
        if self._do_load_key(key):
            try:
                self._cache[key] = self._plugin.get(key)
            except KeyError as e:
                pass
            else:
                self._retrieved[key] = self._cache[key]
        return self._cache.get(key, default)

    def items(self):
        return self._cache.items()

    def values(self):
        return self._cache.values()

    def keys(self):
        return self._cache.keys()

    def pop(self, key, *args):
        if args:
            return self._cache.pop(key, args[0])
        return self._cache.pop(key)

    def __delitem__(self, key):
        del self._cache[key]

    def __setitem__(self, key, value):
        self._cache[key] = value

    def flush(self):
        self._plugin.flush()
        self._cache = {}

    def update(self, value):
        self._cache.update(value)