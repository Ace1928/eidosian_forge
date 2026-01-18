import time
from . import debug, errors, osutils, revision, trace
def _step_unique_and_common_searchers(self, common_searcher, unique_tip_searchers, unique_searcher):
    """Step all the searchers"""
    newly_seen_common = set(common_searcher.step())
    newly_seen_unique = set()
    for searcher in unique_tip_searchers:
        next = set(searcher.step())
        next.update(unique_searcher.find_seen_ancestors(next))
        next.update(common_searcher.find_seen_ancestors(next))
        for alt_searcher in unique_tip_searchers:
            if alt_searcher is searcher:
                continue
            next.update(alt_searcher.find_seen_ancestors(next))
        searcher.start_searching(next)
        newly_seen_unique.update(next)
    return (newly_seen_common, newly_seen_unique)