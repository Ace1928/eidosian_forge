import time
from . import debug, errors, osutils, revision, trace
class _BreadthFirstSearcher:
    """Parallel search breadth-first the ancestry of revisions.

    This class implements the iterator protocol, but additionally
    1. provides a set of seen ancestors, and
    2. allows some ancestries to be unsearched, via stop_searching_any
    """

    def __init__(self, revisions, parents_provider):
        self._iterations = 0
        self._next_query = set(revisions)
        self.seen = set()
        self._started_keys = set(self._next_query)
        self._stopped_keys = set()
        self._parents_provider = parents_provider
        self._returning = 'next_with_ghosts'
        self._current_present = set()
        self._current_ghosts = set()
        self._current_parents = {}

    def __repr__(self):
        if self._iterations:
            prefix = 'searching'
        else:
            prefix = 'starting'
        search = '{}={!r}'.format(prefix, list(self._next_query))
        return '_BreadthFirstSearcher(iterations=%d, %s, seen=%r)' % (self._iterations, search, list(self.seen))

    def get_state(self):
        """Get the current state of this searcher.

        :return: Tuple with started keys, excludes and included keys
        """
        if self._returning == 'next':
            found, ghosts, next, parents = self._do_query(self._next_query)
            self.seen.difference_update(next)
            next_query = next.union(ghosts)
        else:
            next_query = self._next_query
        excludes = self._stopped_keys.union(next_query)
        included_keys = self.seen.difference(excludes)
        return (self._started_keys, excludes, included_keys)

    def step(self):
        try:
            return next(self)
        except StopIteration:
            return ()

    def __next__(self):
        """Return the next ancestors of this revision.

        Ancestors are returned in the order they are seen in a breadth-first
        traversal.  No ancestor will be returned more than once. Ancestors are
        returned before their parentage is queried, so ghosts and missing
        revisions (including the start revisions) are included in the result.
        This can save a round trip in LCA style calculation by allowing
        convergence to be detected without reading the data for the revision
        the convergence occurs on.

        :return: A set of revision_ids.
        """
        if self._returning != 'next':
            self._returning = 'next'
            self._iterations += 1
        else:
            self._advance()
        if len(self._next_query) == 0:
            raise StopIteration()
        self.seen.update(self._next_query)
        return self._next_query
    next = __next__

    def next_with_ghosts(self):
        """Return the next found ancestors, with ghosts split out.

        Ancestors are returned in the order they are seen in a breadth-first
        traversal.  No ancestor will be returned more than once. Ancestors are
        returned only after asking for their parents, which allows us to detect
        which revisions are ghosts and which are not.

        :return: A tuple with (present ancestors, ghost ancestors) sets.
        """
        if self._returning != 'next_with_ghosts':
            self._returning = 'next_with_ghosts'
            self._advance()
        if len(self._next_query) == 0:
            raise StopIteration()
        self._advance()
        return (self._current_present, self._current_ghosts)

    def _advance(self):
        """Advance the search.

        Updates self.seen, self._next_query, self._current_present,
        self._current_ghosts, self._current_parents and self._iterations.
        """
        self._iterations += 1
        found, ghosts, next, parents = self._do_query(self._next_query)
        self._current_present = found
        self._current_ghosts = ghosts
        self._next_query = next
        self._current_parents = parents
        self._stopped_keys.update(ghosts)

    def _do_query(self, revisions):
        """Query for revisions.

        Adds revisions to the seen set.

        :param revisions: Revisions to query.
        :return: A tuple: (set(found_revisions), set(ghost_revisions),
           set(parents_of_found_revisions), dict(found_revisions:parents)).
        """
        found_revisions = set()
        parents_of_found = set()
        seen = self.seen
        seen.update(revisions)
        parent_map = self._parents_provider.get_parent_map(revisions)
        found_revisions.update(parent_map)
        for rev_id, parents in parent_map.items():
            if parents is None:
                continue
            new_found_parents = [p for p in parents if p not in seen]
            if new_found_parents:
                parents_of_found.update(new_found_parents)
        ghost_revisions = revisions - found_revisions
        return (found_revisions, ghost_revisions, parents_of_found, parent_map)

    def __iter__(self):
        return self

    def find_seen_ancestors(self, revisions):
        """Find ancestors of these revisions that have already been seen.

        This function generally makes the assumption that querying for the
        parents of a node that has already been queried is reasonably cheap.
        (eg, not a round trip to a remote host).
        """
        all_seen = self.seen
        pending = set(revisions).intersection(all_seen)
        seen_ancestors = set(pending)
        if self._returning == 'next':
            not_searched_yet = self._next_query
        else:
            not_searched_yet = ()
        pending.difference_update(not_searched_yet)
        get_parent_map = self._parents_provider.get_parent_map
        while pending:
            parent_map = get_parent_map(pending)
            all_parents = []
            for parent_ids in parent_map.values():
                all_parents.extend(parent_ids)
            next_pending = all_seen.intersection(all_parents).difference(seen_ancestors)
            seen_ancestors.update(next_pending)
            next_pending.difference_update(not_searched_yet)
            pending = next_pending
        return seen_ancestors

    def stop_searching_any(self, revisions):
        """
        Remove any of the specified revisions from the search list.

        None of the specified revisions are required to be present in the
        search list.

        It is okay to call stop_searching_any() for revisions which were seen
        in previous iterations. It is the callers responsibility to call
        find_seen_ancestors() to make sure that current search tips that are
        ancestors of those revisions are also stopped.  All explicitly stopped
        revisions will be excluded from the search result's get_keys(), though.
        """
        revisions = frozenset(revisions)
        if self._returning == 'next':
            stopped = self._next_query.intersection(revisions)
            self._next_query = self._next_query.difference(revisions)
        else:
            stopped_present = self._current_present.intersection(revisions)
            stopped = stopped_present.union(self._current_ghosts.intersection(revisions))
            self._current_present.difference_update(stopped)
            self._current_ghosts.difference_update(stopped)
            stop_rev_references = {}
            for rev in stopped_present:
                for parent_id in self._current_parents[rev]:
                    if parent_id not in stop_rev_references:
                        stop_rev_references[parent_id] = 0
                    stop_rev_references[parent_id] += 1
            for parents in self._current_parents.values():
                for parent_id in parents:
                    try:
                        stop_rev_references[parent_id] -= 1
                    except KeyError:
                        pass
            stop_parents = set()
            for rev_id, refs in stop_rev_references.items():
                if refs == 0:
                    stop_parents.add(rev_id)
            self._next_query.difference_update(stop_parents)
        self._stopped_keys.update(stopped)
        self._stopped_keys.update(revisions)
        return stopped

    def start_searching(self, revisions):
        """Add revisions to the search.

        The parents of revisions will be returned from the next call to next()
        or next_with_ghosts(). If next_with_ghosts was the most recently used
        next* call then the return value is the result of looking up the
        ghost/not ghost status of revisions. (A tuple (present, ghosted)).
        """
        revisions = frozenset(revisions)
        self._started_keys.update(revisions)
        new_revisions = revisions.difference(self.seen)
        if self._returning == 'next':
            self._next_query.update(new_revisions)
            self.seen.update(new_revisions)
        else:
            revs, ghosts, query, parents = self._do_query(revisions)
            self._stopped_keys.update(ghosts)
            self._current_present.update(revs)
            self._current_ghosts.update(ghosts)
            self._next_query.update(query)
            self._current_parents.update(parents)
            return (revs, ghosts)