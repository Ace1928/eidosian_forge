import time
from . import debug, errors, osutils, revision, trace
def _collapse_unique_searchers(self, unique_tip_searchers, common_to_all_unique_nodes):
    """Combine searchers that are searching the same tips.

        When two searchers are searching the same tips, we can stop one of the
        searchers. We also know that the maximal set of common ancestors is the
        intersection of the two original searchers.

        :return: A list of searchers that are searching unique nodes.
        """
    unique_search_tips = {}
    for searcher in unique_tip_searchers:
        stopped = searcher.stop_searching_any(common_to_all_unique_nodes)
        will_search_set = frozenset(searcher._next_query)
        if not will_search_set:
            if 'graph' in debug.debug_flags:
                trace.mutter('Unique searcher %s was stopped. (%s iterations) %d nodes stopped', searcher._label, searcher._iterations, len(stopped))
        elif will_search_set not in unique_search_tips:
            unique_search_tips[will_search_set] = [searcher]
        else:
            unique_search_tips[will_search_set].append(searcher)
    next_unique_searchers = []
    for searchers in unique_search_tips.values():
        if len(searchers) == 1:
            next_unique_searchers.append(searchers[0])
        else:
            next_searcher = searchers[0]
            for searcher in searchers[1:]:
                next_searcher.seen.intersection_update(searcher.seen)
            if 'graph' in debug.debug_flags:
                trace.mutter('Combining %d searchers into a single searcher searching %d nodes with %d ancestry', len(searchers), len(next_searcher._next_query), len(next_searcher.seen))
            next_unique_searchers.append(next_searcher)
    return next_unique_searchers