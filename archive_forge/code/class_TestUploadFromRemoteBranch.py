import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestUploadFromRemoteBranch(tests.TestCaseWithTransport, UploadUtilsMixin):
    remote_branch_dir = 'remote_branch'

    def setUp(self):
        super().setUp()
        if self._will_escape_isolation(self.transport_server):
            raise tests.TestNotApplicable('Escaping test isolation')
        self.remote_branch_url = self.make_remote_branch_without_working_tree()

    @staticmethod
    def _will_escape_isolation(transport_server):
        if not features.paramiko.available():
            return False
        from ....tests import stub_sftp
        if transport_server is stub_sftp.SFTPHomeDirServer:
            return True
        return False

    def make_remote_branch_without_working_tree(self):
        """Creates a branch without working tree to upload from.

        It's created from the existing self.branch_dir one which still has its
        working tree.
        """
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        remote_branch_url = self.get_url(self.remote_branch_dir)
        self.run_bzr(['push', remote_branch_url, '--directory', self.branch_dir])
        return remote_branch_url

    def test_no_upload_to_remote_working_tree(self):
        cmd = self._get_cmd_upload()
        up_url = self.get_url(self.branch_dir)
        self.assertRaises(cmds.CannotUploadToWorkingTree, cmd.run, up_url, directory=self.remote_branch_url)

    def test_upload_without_working_tree(self):
        self.do_full_upload(directory=self.remote_branch_url)
        self.assertUpFileEqual(b'foo', 'hello')