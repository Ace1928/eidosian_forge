import time
from . import debug, errors, osutils, revision, trace
def _refine_unique_nodes(self, unique_searcher, all_unique_searcher, unique_tip_searchers, common_searcher):
    """Steps 5-8 of find_unique_ancestors.

        This function returns when common_searcher has stopped searching for
        more nodes.
        """
    step_all_unique_counter = 0
    while common_searcher._next_query:
        newly_seen_common, newly_seen_unique = self._step_unique_and_common_searchers(common_searcher, unique_tip_searchers, unique_searcher)
        common_to_all_unique_nodes = self._find_nodes_common_to_all_unique(unique_tip_searchers, all_unique_searcher, newly_seen_unique, step_all_unique_counter == 0)
        step_all_unique_counter = (step_all_unique_counter + 1) % STEP_UNIQUE_SEARCHER_EVERY
        if newly_seen_common:
            common_searcher.stop_searching_any(all_unique_searcher.seen.intersection(newly_seen_common))
        if common_to_all_unique_nodes:
            common_to_all_unique_nodes.update(common_searcher.find_seen_ancestors(common_to_all_unique_nodes))
            all_unique_searcher.start_searching(common_to_all_unique_nodes)
            common_searcher.stop_searching_any(common_to_all_unique_nodes)
        next_unique_searchers = self._collapse_unique_searchers(unique_tip_searchers, common_to_all_unique_nodes)
        if len(unique_tip_searchers) != len(next_unique_searchers):
            if 'graph' in debug.debug_flags:
                trace.mutter('Collapsed %d unique searchers => %d at %s iterations', len(unique_tip_searchers), len(next_unique_searchers), all_unique_searcher._iterations)
        unique_tip_searchers = next_unique_searchers