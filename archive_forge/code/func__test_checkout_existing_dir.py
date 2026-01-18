import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def _test_checkout_existing_dir(self, lightweight):
    source = self.make_branch_and_tree('source')
    self.build_tree_contents([('source/file1', b'content1'), ('source/file2', b'content2')])
    source.add(['file1', 'file2'])
    source.commit('added files')
    self.build_tree_contents([('target/', b''), ('target/file1', b'content1'), ('target/file2', b'content3')])
    cmd = ['checkout', 'source', 'target']
    if lightweight:
        cmd.append('--lightweight')
    self.run_bzr('checkout source target')
    self.assertPathExists('target/file2.moved')
    self.assertPathDoesNotExist('target/file1.moved')