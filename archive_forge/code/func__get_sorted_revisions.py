import sys
from . import delta as _mod_delta
from . import errors as errors
from . import hooks as _mod_hooks
from . import log, osutils
from . import revision as _mod_revision
from . import tsort
from .trace import mutter, warning
from .workingtree import ShelvingUnsupported
def _get_sorted_revisions(tip_revision, revision_ids, parent_map):
    """Get an iterator which will return the revisions in merge sorted order.

    This will build up a list of all nodes, such that only nodes in the list
    are referenced. It then uses MergeSorter to return them in 'merge-sorted'
    order.

    :param revision_ids: A set of revision_ids
    :param parent_map: The parent information for each node. Revisions which
        are considered ghosts should not be present in the map.
    :return: iterator from MergeSorter.iter_topo_order()
    """
    parent_graph = {}
    for revision_id in revision_ids:
        if revision_id not in parent_map:
            parent_graph[revision_id] = []
        else:
            parent_graph[revision_id] = [p for p in parent_map[revision_id] if p in revision_ids]
    sorter = tsort.MergeSorter(parent_graph, tip_revision)
    return sorter.iter_topo_order()