import time
from . import debug, errors, osutils, revision, trace
class StackedParentsProvider:
    """A parents provider which stacks (or unions) multiple providers.

    The providers are queries in the order of the provided parent_providers.
    """

    def __init__(self, parent_providers):
        self._parent_providers = parent_providers

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._parent_providers)

    def get_parent_map(self, keys):
        """Get a mapping of keys => parents

        A dictionary is returned with an entry for each key present in this
        source. If this source doesn't have information about a key, it should
        not include an entry.

        [NULL_REVISION] is used as the parent of the first user-committed
        revision.  Its parent list is empty.

        :param keys: An iterable returning keys to check (eg revision_ids)
        :return: A dictionary mapping each key to its parents
        """
        found = {}
        remaining = set(keys)
        for parents_provider in self._parent_providers:
            get_cached = getattr(parents_provider, 'get_cached_parent_map', None)
            if get_cached is None:
                continue
            new_found = get_cached(remaining)
            found.update(new_found)
            remaining.difference_update(new_found)
            if not remaining:
                break
        if not remaining:
            return found
        for parents_provider in self._parent_providers:
            try:
                new_found = parents_provider.get_parent_map(remaining)
            except errors.UnsupportedOperation:
                continue
            found.update(new_found)
            remaining.difference_update(new_found)
            if not remaining:
                break
        return found