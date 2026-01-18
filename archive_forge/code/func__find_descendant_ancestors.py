import time
from . import debug, errors, osutils, revision, trace
def _find_descendant_ancestors(self, old_key, new_key):
    """Find ancestors of new_key that may be descendants of old_key."""
    stop = self._make_breadth_first_searcher([old_key])
    descendants = self._make_breadth_first_searcher([new_key])
    for revisions in descendants:
        old_stop = stop.seen.intersection(revisions)
        descendants.stop_searching_any(old_stop)
        seen_stop = descendants.find_seen_ancestors(stop.step())
        descendants.stop_searching_any(seen_stop)
    return descendants.seen.difference(stop.seen)