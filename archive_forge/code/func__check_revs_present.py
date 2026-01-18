from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def _check_revs_present(self, br2):
    for rev_id in [b'1-1', b'1-2', b'2-1']:
        self.assertTrue(br2.repository.has_revision(rev_id))
        rev = br2.repository.get_revision(rev_id)
        self.assertEqual(rev.revision_id, rev_id)
        self.assertTrue(br2.repository.get_inventory(rev_id))