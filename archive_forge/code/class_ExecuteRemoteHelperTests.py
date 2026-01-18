import os
import subprocess
import sys
from io import BytesIO
from dulwich.repo import Repo
from ...tests import TestCaseWithTransport
from ...tests.features import PathFeature
from ..git_remote_helper import RemoteHelper, fetch, open_local_dir
from ..object_store import get_object_store
from . import FastimportFeature
class ExecuteRemoteHelperTests(TestCaseWithTransport):

    def test_run(self):
        self.requireFeature(git_remote_bzr_feature)
        local_dir = self.make_branch_and_tree('local', format='git').controldir
        local_path = local_dir.control_transport.local_abspath('.')
        remote_tree = self.make_branch_and_tree('remote')
        remote_dir = remote_tree.controldir
        shortname = 'bzr'
        env = dict(os.environ)
        env['GIT_DIR'] = local_path
        env['PYTHONPATH'] = ':'.join(sys.path)
        p = subprocess.Popen([sys.executable, git_remote_bzr_path, local_path, remote_dir.user_url], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        out, err = p.communicate(b'capabilities\n')
        lines = out.splitlines()
        self.assertIn(b'push', lines, "no 'push' in {!r}, error: {!r}".format(lines, err))
        self.assertEqual(b"git-remote-bzr is experimental and has not been optimized for performance. Use 'brz fast-export' and 'git fast-import' for large repositories.\n", err)