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
class WorkingTreeRevisionRewriter:

    def __init__(self, wt, state, merge_type=None):
        """
        :param wt: Working tree in which to do the replays.
        """
        self.wt = wt
        self.graph = self.wt.branch.repository.get_graph()
        self.state = state
        self.merge_type = merge_type

    def __call__(self, oldrevid, newrevid, newparents):
        """Replay a commit in a working tree, with a different base.

        :param oldrevid: Old revision id
        :param newrevid: New revision id
        :param newparents: New parent revision ids
        """
        repository = self.wt.branch.repository
        if self.merge_type is None:
            from ...merge import Merge3Merger
            merge_type = Merge3Merger
        else:
            merge_type = self.merge_type
        oldrev = self.wt.branch.repository.get_revision(oldrevid)
        complete_revert(self.wt, [newparents[0]])
        assert not self.wt.changes_from(self.wt.basis_tree()).has_changed(), 'Changes in rev'
        oldtree = repository.revision_tree(oldrevid)
        self.state.write_active_revid(oldrevid)
        merger = Merger(self.wt.branch, this_tree=self.wt)
        merger.set_other_revision(oldrevid, self.wt.branch)
        base_revid = self.determine_base(oldrevid, oldrev.parent_ids, newrevid, newparents)
        mutter('replaying %r as %r with base %r and new parents %r' % (oldrevid, newrevid, base_revid, newparents))
        merger.set_base_revision(base_revid, self.wt.branch)
        merger.merge_type = merge_type
        merger.do_merge()
        for newparent in newparents[1:]:
            self.wt.add_pending_merge(newparent)
        self.commit_rebase(oldrev, newrevid)
        self.state.write_active_revid(None)

    def determine_base(self, oldrevid, oldparents, newrevid, newparents):
        """Determine the base for replaying a revision using merge.

        :param oldrevid: Revid of old revision.
        :param oldparents: List of old parents revids.
        :param newrevid: Revid of new revision.
        :param newparents: List of new parents revids.
        :return: Revision id of the new new revision.
        """
        if len(oldparents) == 0:
            return NULL_REVISION
        if len(oldparents) == 1:
            return oldparents[0]
        if len(newparents) == 1:
            return oldparents[1]
        try:
            return self.graph.find_unique_lca(*[oldparents[0], newparents[1]])
        except NoCommonAncestor:
            return oldparents[0]

    def commit_rebase(self, oldrev, newrevid):
        """Commit a rebase.

        :param oldrev: Revision info of new revision to commit.
        :param newrevid: New revision id."""
        assert oldrev.revision_id != newrevid, 'Invalid revid %r' % newrevid
        revprops = dict(oldrev.properties)
        revprops[REVPROP_REBASE_OF] = oldrev.revision_id.decode('utf-8')
        committer = self.wt.branch.get_config().username()
        authors = oldrev.get_apparent_authors()
        if oldrev.committer == committer:
            if [oldrev.committer] == authors:
                authors = None
        elif oldrev.committer not in authors:
            authors.append(oldrev.committer)
        if 'author' in revprops:
            del revprops['author']
        if 'authors' in revprops:
            del revprops['authors']
        self.wt.commit(message=oldrev.message, timestamp=oldrev.timestamp, timezone=oldrev.timezone, revprops=revprops, rev_id=newrevid, committer=committer, authors=authors)