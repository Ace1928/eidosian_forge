import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestUploadBadRemoteReivd(tests.TestCaseWithTransport, UploadUtilsMixin):

    def test_raises_on_wrong_revid(self):
        tree = self.make_branch_and_working_tree()
        self.do_full_upload()
        t = self.get_transport(self.upload_dir)
        t.put_bytes('.bzr-upload.revid', b'fake')
        self.add_file('foo', b'bar\n')
        self.assertRaises(cmds.DivergedUploadedTree, self.do_full_upload)