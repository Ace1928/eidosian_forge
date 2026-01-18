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
class TestFormat5(TestCaseWithTransport):
    """Tests specific to the version 5 bzrdir format."""

    def test_same_lockfiles_between_tree_repo_branch(self):
        dir = BzrDirFormat5().initialize(self.get_url())

        def check_dir_components_use_same_lock(dir):
            ctrl_1 = dir.open_repository().control_files
            ctrl_2 = dir.open_branch().control_files
            ctrl_3 = dir.open_workingtree()._control_files
            self.assertTrue(ctrl_1 is ctrl_2)
            self.assertTrue(ctrl_2 is ctrl_3)
        check_dir_components_use_same_lock(dir)
        dir = controldir.ControlDir.open(self.get_url())
        check_dir_components_use_same_lock(dir)

    def test_can_convert(self):
        dir = BzrDirFormat5().initialize(self.get_url())
        self.assertTrue(dir.can_convert_format())

    def test_needs_conversion(self):
        dir = BzrDirFormat5().initialize(self.get_url())
        self.assertFalse(dir.needs_format_conversion(BzrDirFormat5()))
        self.assertTrue(dir.needs_format_conversion(bzrdir.BzrDirFormat.get_default_format()))