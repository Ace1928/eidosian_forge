import time
from . import debug, errors, osutils, revision, trace
def _find_initial_unique_nodes(self, unique_revisions, common_revisions):
    """Steps 1-3 of find_unique_ancestors.

        Find the maximal set of unique nodes. Some of these might actually
        still be common, but we are sure that there are no other unique nodes.

        :return: (unique_searcher, common_searcher)
        """
    unique_searcher = self._make_breadth_first_searcher(unique_revisions)
    next(unique_searcher)
    common_searcher = self._make_breadth_first_searcher(common_revisions)
    while unique_searcher._next_query:
        next_unique_nodes = set(unique_searcher.step())
        next_common_nodes = set(common_searcher.step())
        unique_are_common_nodes = next_unique_nodes.intersection(common_searcher.seen)
        unique_are_common_nodes.update(next_common_nodes.intersection(unique_searcher.seen))
        if unique_are_common_nodes:
            ancestors = unique_searcher.find_seen_ancestors(unique_are_common_nodes)
            ancestors.update(common_searcher.find_seen_ancestors(ancestors))
            unique_searcher.stop_searching_any(ancestors)
            common_searcher.start_searching(ancestors)
    return (unique_searcher, common_searcher)