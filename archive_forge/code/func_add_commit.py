import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
def add_commit(repo, revision_id, parent_ids):
    repo.lock_write()
    repo.start_write_group()
    inv = Inventory(revision_id=revision_id)
    inv.root.revision = revision_id
    root_id = inv.root.file_id
    sha1 = repo.add_inventory(revision_id, inv, parent_ids)
    repo.texts.add_lines((root_id, revision_id), [], [])
    rev = breezy.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
    rev.parent_ids = parent_ids
    repo.add_revision(revision_id, rev)
    repo.commit_write_group()
    repo.unlock()