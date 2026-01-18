import os
import tempfile
from breezy import osutils, tests, transport, urlutils
class TestOutsideWT(tests.ChrootedTestCase):
    """Test that brz gives proper errors outside of a working tree."""

    def test_cwd_log(self):
        tmp_dir = osutils.realpath(tempfile.mkdtemp())
        self.permit_url('file:///')
        self.addCleanup(osutils.rmtree, tmp_dir)
        out, err = self.run_bzr('log', retcode=3, working_dir=tmp_dir)
        self.assertEqual('brz: ERROR: Not a branch: "%s/".\n' % (tmp_dir,), err)

    def test_url_log(self):
        url = self.get_readonly_url() + 'subdir/'
        out, err = self.run_bzr(['log', url], retcode=3)
        self.assertEqual('brz: ERROR: Not a branch: "%s".\n' % url, err)

    def test_diff_outside_tree(self):
        tree = self.make_branch_and_tree('branch1')
        tree.commit('nothing')
        tree.commit('nothing')
        tmp_dir = osutils.realpath(tempfile.mkdtemp())
        self.addCleanup(osutils.rmtree, tmp_dir)
        self.permit_url('file:///')
        expected_error = 'brz: ERROR: Not a branch: "%s/branch2/".\n' % tmp_dir
        out, err = self.run_bzr('diff -r revno:2:branch2..revno:1', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        out, err = self.run_bzr('diff -r revno:2:branch2', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        out, err = self.run_bzr('diff -r revno:2:branch2..', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        out, err = self.run_bzr('diff', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual('brz: ERROR: Not a branch: "%s/".\n' % tmp_dir, err)