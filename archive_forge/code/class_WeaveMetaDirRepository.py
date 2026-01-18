import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
class WeaveMetaDirRepository(MetaDirVersionedFileRepository):
    """A subclass of MetaDirRepository to set weave specific policy."""

    def __init__(self, _format, a_controldir, control_files):
        super().__init__(_format, a_controldir, control_files)
        self._serializer = _format._serializer

    def _all_possible_ids(self):
        """Return all the possible revisions that we could find."""
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, '_all_possible_ids scales with size of history.')
        with self.lock_read():
            return [key[-1] for key in self.inventories.keys()]

    def _all_revision_ids(self):
        """Returns a list of all the revision ids in the repository.

        These are in as much topological order as the underlying store can
        present: for weaves ghosts may lead to a lack of correctness until
        the reweave updates the parents list.
        """
        with self.lock_read():
            return [key[-1] for key in self.revisions.keys()]

    def _activate_new_inventory(self):
        """Put a replacement inventory.new into use as inventories."""
        t = self._transport
        t.copy('inventory.new.weave', 'inventory.weave')
        t.delete('inventory.new.weave')
        self.inventories.keys()

    def _backup_inventory(self):
        t = self._transport
        t.copy('inventory.weave', 'inventory.backup.weave')

    def _temp_inventories(self):
        t = self._transport
        return self._format._get_inventories(t, self, 'inventory.new')

    def get_commit_builder(self, branch, parents, config, timestamp=None, timezone=None, committer=None, revprops=None, revision_id=None, lossy=False):
        self._check_ascii_revisionid(revision_id, self.get_commit_builder)
        result = VersionedFileCommitBuilder(self, parents, config, timestamp, timezone, committer, revprops, revision_id, lossy=lossy)
        self.start_write_group()
        return result

    def get_revision(self, revision_id):
        """Return the Revision object for a named revision"""
        with self.lock_read():
            return self.get_revision_reconcile(revision_id)

    def _inventory_add_lines(self, revision_id, parents, lines, check_content=True):
        """Store lines in inv_vf and return the sha1 of the inventory."""
        present_parents = self.get_graph().get_parent_map(parents)
        final_parents = []
        for parent in parents:
            if parent in present_parents:
                final_parents.append((parent,))
        return self.inventories.add_lines((revision_id,), final_parents, lines, check_content=check_content)[0]