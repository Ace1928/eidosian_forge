import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
class TestEscaping(tests.TestCaseWithTransport):
    """Test that repositories can be stored correctly on VFAT transports.

    Makes sure we have proper escaping of invalid characters, etc.

    It'd be better to test all operations on the FakeVFATTransportDecorator,
    but working trees go straight to the os not through the Transport layer.
    Therefore we build some history first in the regular way and then
    check it's safe to access for vfat.
    """

    def test_on_vfat(self):
        if isinstance(self.repository_format, remote.RemoteRepositoryFormat):
            return
        self.transport_server = test_server.FakeVFATServer
        FOO_ID = b'foo<:>ID'
        wt = self.make_branch_and_tree('repo')
        if not wt.supports_setting_file_ids():
            self.skip('format does not support setting file ids')
        self.build_tree(['repo/foo'], line_endings='binary')
        wt.add(['foo'], ids=[FOO_ID])
        rev1 = wt.commit('this is my new commit')
        branch = controldir.ControlDir.open(self.get_url('repo')).open_branch()
        revtree = branch.repository.revision_tree(rev1)
        revtree.lock_read()
        self.addCleanup(revtree.unlock)
        contents = revtree.get_file_text('foo')
        self.assertEqual(contents, b'contents of repo/foo\n')

    def test_create_bundle(self):
        wt = self.make_branch_and_tree('repo')
        self.build_tree(['repo/file1'])
        wt.add('file1')
        rev1 = wt.commit('file1')
        fileobj = BytesIO()
        wt.branch.repository.create_bundle(rev1, _mod_revision.NULL_REVISION, fileobj)