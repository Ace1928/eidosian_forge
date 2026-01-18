from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
class WordsetDict(MutableMapping):
    """
    Special mapping used to store dictionary of wordsets.
    Different from a regular dict in that some wordsets
    may be lazy-loaded from an asset path.
    """
    paths = None
    _loaded = None

    def __init__(self, *args, **kwds):
        self.paths = {}
        self._loaded = {}
        super(WordsetDict, self).__init__(*args, **kwds)

    def __getitem__(self, key):
        try:
            return self._loaded[key]
        except KeyError:
            pass
        path = self.paths[key]
        value = self._loaded[key] = _load_wordset(path)
        return value

    def set_path(self, key, path):
        """
        set asset path to lazy-load wordset from.
        """
        self.paths[key] = path

    def __setitem__(self, key, value):
        self._loaded[key] = value

    def __delitem__(self, key):
        if key in self:
            del self._loaded[key]
            self.paths.pop(key, None)
        else:
            del self.paths[key]

    @property
    def _keyset(self):
        keys = set(self._loaded)
        keys.update(self.paths)
        return keys

    def __iter__(self):
        return iter(self._keyset)

    def __len__(self):
        return len(self._keyset)

    def __contains__(self, key):
        return key in self._loaded or key in self.paths