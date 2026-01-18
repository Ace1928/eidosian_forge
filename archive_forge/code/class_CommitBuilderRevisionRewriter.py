import os
from ... import config as _mod_config
from ... import osutils, ui
from ...bzr.generate_ids import gen_revision_id
from ...bzr.inventorytree import InventoryTreeChange
from ...errors import (BzrError, NoCommonAncestor, UnknownFormatError,
from ...graph import FrozenHeadsCache
from ...merge import Merger
from ...revision import NULL_REVISION
from ...trace import mutter
from ...transport import NoSuchFile
from ...tsort import topo_sort
from .maptree import MapTree, map_file_ids
class CommitBuilderRevisionRewriter:
    """Revision rewriter that use commit builder.

    :ivar repository: Repository in which the revision is present.
    """

    def __init__(self, repository, map_ids=True):
        self.repository = repository
        self.map_ids = map_ids

    def _get_present_revisions(self, revids):
        return tuple([p for p in revids if self.repository.has_revision(p)])

    def __call__(self, oldrevid, newrevid, new_parents):
        """Replay a commit by simply commiting the same snapshot with different
        parents.

        :param oldrevid: Revision id of the revision to copy.
        :param newrevid: Revision id of the revision to create.
        :param new_parents: Revision ids of the new parent revisions.
        """
        assert isinstance(new_parents, tuple), 'CommitBuilderRevisionRewriter: Expected tuple for %r' % new_parents
        mutter('creating copy %r of %r with new parents %r', newrevid, oldrevid, new_parents)
        oldrev = self.repository.get_revision(oldrevid)
        revprops = dict(oldrev.properties)
        revprops[REVPROP_REBASE_OF] = oldrevid.decode('utf-8')
        nonghost_oldparents = self._get_present_revisions(oldrev.parent_ids)
        nonghost_newparents = self._get_present_revisions(new_parents)
        oldtree = self.repository.revision_tree(oldrevid)
        if self.map_ids:
            fileid_map = map_file_ids(self.repository, nonghost_oldparents, nonghost_newparents)
            mappedtree = MapTree(oldtree, fileid_map)
        else:
            mappedtree = oldtree
        try:
            old_base = nonghost_oldparents[0]
        except IndexError:
            old_base = NULL_REVISION
        try:
            new_base = new_parents[0]
        except IndexError:
            new_base = NULL_REVISION
        old_base_tree = self.repository.revision_tree(old_base)
        old_iter_changes = oldtree.iter_changes(old_base_tree)
        iter_changes = wrap_iter_changes(old_iter_changes, mappedtree)
        builder = self.repository.get_commit_builder(branch=None, parents=new_parents, committer=oldrev.committer, timestamp=oldrev.timestamp, timezone=oldrev.timezone, revprops=revprops, revision_id=newrevid, config_stack=_mod_config.GlobalStack())
        try:
            for relpath, fs_hash in builder.record_iter_changes(mappedtree, new_base, iter_changes):
                pass
            builder.finish_inventory()
            return builder.commit(oldrev.message)
        except:
            builder.abort()
            raise