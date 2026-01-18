import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
class TestLossyPush(per_branch.TestCaseWithBranch):

    def setUp(self):
        self.hook_calls = []
        super().setUp()

    def test_lossy_push_raises_same_vcs(self):
        target = self.make_branch('target')
        source = self.make_branch('source')
        self.assertRaises(errors.LossyPushToSameVCS, source.push, target, lossy=True)