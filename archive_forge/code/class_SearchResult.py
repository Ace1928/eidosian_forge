import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class SearchResult(AbstractSearchResult):
    """The result of a breadth first search.

    A SearchResult provides the ability to reconstruct the search or access a
    set of the keys the search found.
    """

    def __init__(self, start_keys, exclude_keys, key_count, keys):
        """Create a SearchResult.

        :param start_keys: The keys the search started at.
        :param exclude_keys: The keys the search excludes.
        :param key_count: The total number of keys (from start to but not
            including exclude).
        :param keys: The keys the search found. Note that in future we may get
            a SearchResult from a smart server, in which case the keys list is
            not necessarily immediately available.
        """
        self._recipe = ('search', start_keys, exclude_keys, key_count)
        self._keys = frozenset(keys)

    def __repr__(self):
        kind, start_keys, exclude_keys, key_count = self._recipe
        if len(start_keys) > 5:
            start_keys_repr = repr(list(start_keys)[:5])[:-1] + ', ...]'
        else:
            start_keys_repr = repr(start_keys)
        if len(exclude_keys) > 5:
            exclude_keys_repr = repr(list(exclude_keys)[:5])[:-1] + ', ...]'
        else:
            exclude_keys_repr = repr(exclude_keys)
        return '<%s %s:(%s, %s, %d)>' % (self.__class__.__name__, kind, start_keys_repr, exclude_keys_repr, key_count)

    def get_recipe(self):
        """Return a recipe that can be used to replay this search.

        The recipe allows reconstruction of the same results at a later date
        without knowing all the found keys. The essential elements are a list
        of keys to start and to stop at. In order to give reproducible
        results when ghosts are encountered by a search they are automatically
        added to the exclude list (or else ghost filling may alter the
        results).

        :return: A tuple ('search', start_keys_set, exclude_keys_set,
            revision_count). To recreate the results of this search, create a
            breadth first searcher on the same graph starting at start_keys.
            Then call next() (or next_with_ghosts()) repeatedly, and on every
            result, call stop_searching_any on any keys from the exclude_keys
            set. The revision_count value acts as a trivial cross-check - the
            found revisions of the new search should have as many elements as
            revision_count. If it does not, then additional revisions have been
            ghosted since the search was executed the first time and the second
            time.
        """
        return self._recipe

    def get_network_struct(self):
        start_keys = b' '.join(self._recipe[1])
        stop_keys = b' '.join(self._recipe[2])
        count = str(self._recipe[3]).encode('ascii')
        return (self._recipe[0].encode('ascii'), b'\n'.join((start_keys, stop_keys, count)))

    def get_keys(self):
        """Return the keys found in this search.

        :return: A set of keys.
        """
        return self._keys

    def is_empty(self):
        """Return false if the search lists 1 or more revisions."""
        return self._recipe[3] == 0

    def refine(self, seen, referenced):
        """Create a new search by refining this search.

        :param seen: Revisions that have been satisfied.
        :param referenced: Revision references observed while satisfying some
            of this search.
        """
        start = self._recipe[1]
        exclude = self._recipe[2]
        count = self._recipe[3]
        keys = self.get_keys()
        pending_refs = set(referenced)
        pending_refs.update(start)
        pending_refs.difference_update(seen)
        pending_refs.difference_update(exclude)
        seen_heads = start.intersection(seen)
        exclude.update(seen_heads)
        keys = keys - seen
        count -= len(seen)
        return SearchResult(pending_refs, exclude, count, keys)