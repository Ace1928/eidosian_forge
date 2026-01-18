import time
from . import debug, errors, osutils, revision, trace
def _remove_simple_descendants(self, revisions, parent_map):
    """remove revisions which are children of other ones in the set

        This doesn't do any graph searching, it just checks the immediate
        parent_map to find if there are any children which can be removed.

        :param revisions: A set of revision_ids
        :return: A set of revision_ids with the children removed
        """
    simple_ancestors = revisions.copy()
    for revision, parent_ids in parent_map.items():
        if parent_ids is None:
            continue
        for parent_id in parent_ids:
            if parent_id in revisions:
                simple_ancestors.discard(revision)
                break
    return simple_ancestors