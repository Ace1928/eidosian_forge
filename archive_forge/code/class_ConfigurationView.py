import time
from collections import OrderedDict as _OrderedDict
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, MutableSet, Sequence
from heapq import heapify, heappop, heappush
from itertools import chain, count
from queue import Empty
from typing import Any, Dict, Iterable, List  # noqa
from .functional import first, uniq
from .text import match_case
class ConfigurationView(ChainMap, AttributeDictMixin):
    """A view over an applications configuration dictionaries.

    Custom (but older) version of :class:`collections.ChainMap`.

    If the key does not exist in ``changes``, the ``defaults``
    dictionaries are consulted.

    Arguments:
        changes (Mapping): Map of configuration changes.
        defaults (List[Mapping]): List of dictionaries containing
            the default configuration.
    """

    def __init__(self, changes, defaults=None, keys=None, prefix=None):
        defaults = [] if defaults is None else defaults
        super().__init__(changes, *defaults)
        self.__dict__.update(prefix=prefix.rstrip('_') + '_' if prefix else prefix, _keys=keys)

    def _to_keys(self, key):
        prefix = self.prefix
        if prefix:
            pkey = prefix + key if not key.startswith(prefix) else key
            return (match_case(pkey, prefix), key)
        return (key,)

    def __getitem__(self, key):
        keys = self._to_keys(key)
        getitem = super().__getitem__
        for k in keys + (tuple((f(key) for f in self._keys)) if self._keys else ()):
            try:
                return getitem(k)
            except KeyError:
                pass
        try:
            return self.__missing__(key)
        except KeyError:
            if len(keys) > 1:
                raise KeyError('Key not found: {0!r} (with prefix: {0!r})'.format(*keys))
            raise

    def __setitem__(self, key, value):
        self.changes[self._key(key)] = value

    def first(self, *keys):
        return first(None, (self.get(key) for key in keys))

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        """Remove all changes, but keep defaults."""
        self.changes.clear()

    def __contains__(self, key):
        keys = self._to_keys(key)
        return any((any((k in m for k in keys)) for m in self.maps))

    def swap_with(self, other):
        changes = other.__dict__['changes']
        defaults = other.__dict__['defaults']
        self.__dict__.update(changes=changes, defaults=defaults, key_t=other.__dict__['key_t'], prefix=other.__dict__['prefix'], maps=[changes] + defaults)