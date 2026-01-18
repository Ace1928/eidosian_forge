import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def _find_root_ids(self, revs, parent_map, graph):
    revision_root = {}
    for tree in self.iter_rev_trees(revs):
        root_id = tree.path2id('')
        revision_id = tree.get_file_revision('')
        revision_root[revision_id] = root_id
    parents = set(parent_map.values())
    parents.difference_update(revision_root)
    parents.discard(NULL_REVISION)
    parents = graph.get_parent_map(parents)
    for tree in self.iter_rev_trees(parents):
        root_id = tree.path2id('')
        revision_root[tree.get_revision_id()] = root_id
    return revision_root