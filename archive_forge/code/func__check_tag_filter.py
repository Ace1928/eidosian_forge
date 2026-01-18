from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def _check_tag_filter(self, argstr, expected_revnos):
    out, err = self.run_bzr('tags ' + argstr)
    self.assertEqual(err, '')
    self.assertContainsRe(out, '^' + ''.join(['tag {} +{}\n'.format(revno, revno) for revno in expected_revnos]) + '$')