from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class ForeignVcsRegistryTests(tests.TestCase):
    """Tests for the ForeignVcsRegistry class."""

    def test_parse_revision_id_no_dash(self):
        reg = foreign.ForeignVcsRegistry()
        self.assertRaises(errors.InvalidRevisionId, reg.parse_revision_id, b'invalid')

    def test_parse_revision_id_unknown_mapping(self):
        reg = foreign.ForeignVcsRegistry()
        self.assertRaises(errors.InvalidRevisionId, reg.parse_revision_id, b'unknown-foreignrevid')

    def test_parse_revision_id(self):
        reg = foreign.ForeignVcsRegistry()
        vcs = DummyForeignVcs()
        reg.register('dummy', vcs, 'Dummy VCS')
        self.assertEqual(((b'some', b'foreign', b'revid'), DummyForeignVcsMapping(vcs)), reg.parse_revision_id(b'dummy-v1:some-foreign-revid'))