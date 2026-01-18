import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
class TestInit(TestCaseWithTransport):

    def test_init_weave(self):
        out, err = self.run_bzr('init --format=weave')
        self.assertEqual('Created a standalone tree (format: weave)\n', out)
        self.assertEqual('', err)