import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
class Inter1and2Helper:
    """Helper for operations that convert data from model 1 and 2

    This is for use by fetchers and converters.
    """
    known_graph_threshold = 100

    def __init__(self, source):
        """Constructor.

        Args:
          source: The repository data comes from
        """
        self.source = source

    def iter_rev_trees(self, revs):
        """Iterate through RevisionTrees efficiently.

        Additionally, the inventory's revision_id is set if unset.

        Trees are retrieved in batches of 100, and then yielded in the order
        they were requested.

        Args:
          revs: A list of revision ids
        """
        revs = list(revs)
        while revs:
            for tree in self.source.revision_trees(revs[:100]):
                if tree.root_inventory.revision_id is None:
                    tree.root_inventory.revision_id = tree.get_revision_id()
                yield tree
            revs = revs[100:]

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

    def generate_root_texts(self, revs):
        """Generate VersionedFiles for all root ids.

        Args:
          revs: the revisions to include
        """
        from ..tsort import topo_sort
        graph = self.source.get_graph()
        parent_map = graph.get_parent_map(revs)
        rev_order = topo_sort(parent_map)
        rev_id_to_root_id = self._find_root_ids(revs, parent_map, graph)
        root_id_order = [(rev_id_to_root_id[rev_id], rev_id) for rev_id in rev_order]
        root_id_order.sort(key=operator.itemgetter(0))
        if len(revs) > self.known_graph_threshold:
            graph = self.source.get_known_graph_ancestry(revs)
        new_roots_stream = _new_root_data_stream(root_id_order, rev_id_to_root_id, parent_map, self.source, graph)
        return [('texts', new_roots_stream)]