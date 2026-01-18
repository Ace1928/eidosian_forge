import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def _parent_keys_for_root_version(root_id, rev_id, rev_id_to_root_id_map, parent_map, repo, graph=None):
    """Get the parent keys for a given root id.

    A helper function for _new_root_data_stream.
    """
    rev_parents = parent_map[rev_id]
    parent_ids = []
    for parent_id in rev_parents:
        if parent_id == NULL_REVISION:
            continue
        if parent_id not in rev_id_to_root_id_map:
            try:
                tree = repo.revision_tree(parent_id)
            except errors.NoSuchRevision:
                parent_root_id = None
            else:
                parent_root_id = tree.path2id('')
            rev_id_to_root_id_map[parent_id] = None
        else:
            parent_root_id = rev_id_to_root_id_map[parent_id]
        if root_id == parent_root_id:
            parent_ids.append(parent_id)
        else:
            try:
                tree = repo.revision_tree(parent_id)
            except errors.NoSuchRevision:
                pass
            else:
                try:
                    parent_ids.append(tree.get_file_revision(tree.id2path(root_id, recurse='none')))
                except errors.NoSuchId:
                    pass
    if graph is None:
        graph = repo.get_graph()
    heads = graph.heads(parent_ids)
    selected_ids = []
    for parent_id in parent_ids:
        if parent_id in heads and parent_id not in selected_ids:
            selected_ids.append(parent_id)
    parent_keys = [(root_id, parent_id) for parent_id in selected_ids]
    return parent_keys