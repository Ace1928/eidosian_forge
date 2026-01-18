import time
from . import debug, errors, osutils, revision, trace
class CachingParentsProvider:
    """A parents provider which will cache the revision => parents as a dict.

    This is useful for providers which have an expensive look up.

    Either a ParentsProvider or a get_parent_map-like callback may be
    supplied.  If it provides extra un-asked-for parents, they will be cached,
    but filtered out of get_parent_map.

    The cache is enabled by default, but may be disabled and re-enabled.
    """

    def __init__(self, parent_provider=None, get_parent_map=None):
        """Constructor.

        :param parent_provider: The ParentProvider to use.  It or
            get_parent_map must be supplied.
        :param get_parent_map: The get_parent_map callback to use.  It or
            parent_provider must be supplied.
        """
        self._real_provider = parent_provider
        if get_parent_map is None:
            self._get_parent_map = self._real_provider.get_parent_map
        else:
            self._get_parent_map = get_parent_map
        self._cache = None
        self.enable_cache(True)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._real_provider)

    def enable_cache(self, cache_misses=True):
        """Enable cache."""
        if self._cache is not None:
            raise AssertionError('Cache enabled when already enabled.')
        self._cache = {}
        self._cache_misses = cache_misses
        self.missing_keys = set()

    def disable_cache(self):
        """Disable and clear the cache."""
        self._cache = None
        self._cache_misses = None
        self.missing_keys = set()

    def get_cached_map(self):
        """Return any cached get_parent_map values."""
        if self._cache is None:
            return None
        return dict(self._cache)

    def get_cached_parent_map(self, keys):
        """Return items from the cache.

        This returns the same info as get_parent_map, but explicitly does not
        invoke the supplied ParentsProvider to search for uncached values.
        """
        cache = self._cache
        if cache is None:
            return {}
        return {key: cache[key] for key in keys if key in cache}

    def get_parent_map(self, keys):
        """See StackedParentsProvider.get_parent_map."""
        cache = self._cache
        if cache is None:
            cache = self._get_parent_map(keys)
        else:
            needed_revisions = {key for key in keys if key not in cache}
            needed_revisions.difference_update(self.missing_keys)
            if needed_revisions:
                parent_map = self._get_parent_map(needed_revisions)
                cache.update(parent_map)
                if self._cache_misses:
                    for key in needed_revisions:
                        if key not in parent_map:
                            self.note_missing_key(key)
        result = {}
        for key in keys:
            value = cache.get(key)
            if value is not None:
                result[key] = value
        return result

    def note_missing_key(self, key):
        """Note that key is a missing key."""
        if self._cache_misses:
            self.missing_keys.add(key)