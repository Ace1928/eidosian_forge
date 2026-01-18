import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
class TestGitDir(tests.TestCaseInTempDir):

    def test_get_head_branch_reference(self):
        GitRepo.init('.')
        gd = controldir.ControlDir.open('.')
        self.assertEqual('%s,branch=master' % urlutils.local_path_to_url(os.path.abspath('.')), gd.get_branch_reference())

    def test_get_reference_loop(self):
        r = GitRepo.init('.')
        r.refs.set_symbolic_ref(b'refs/heads/loop', b'refs/heads/loop')
        gd = controldir.ControlDir.open('.')
        self.assertRaises(controldir.BranchReferenceLoop, gd.get_branch_reference, name='loop')

    def test_open_reference_loop(self):
        r = GitRepo.init('.')
        r.refs.set_symbolic_ref(b'refs/heads/loop', b'refs/heads/loop')
        gd = controldir.ControlDir.open('.')
        self.assertRaises(controldir.BranchReferenceLoop, gd.open_branch, name='loop')

    def test_open_existing(self):
        GitRepo.init('.')
        gd = controldir.ControlDir.open('.')
        self.assertIsInstance(gd, dir.LocalGitDir)

    def test_open_ref_parent(self):
        r = GitRepo.init('.')
        cid = r.do_commit(message=b'message', ref=b'refs/heads/foo/bar')
        gd = controldir.ControlDir.open('.')
        self.assertRaises(errors.NotBranchError, gd.open_branch, 'foo')

    def test_open_workingtree(self):
        r = GitRepo.init('.')
        r.do_commit(message=b'message')
        gd = controldir.ControlDir.open('.')
        wt = gd.open_workingtree()
        self.assertIsInstance(wt, workingtree.GitWorkingTree)

    def test_open_workingtree_bare(self):
        GitRepo.init_bare('.')
        gd = controldir.ControlDir.open('.')
        self.assertRaises(errors.NoWorkingTree, gd.open_workingtree)

    def test_git_file(self):
        gitrepo = GitRepo.init('blah', mkdir=True)
        self.build_tree_contents([('foo/',), ('foo/.git', b'gitdir: ../blah/.git\n')])
        gd = controldir.ControlDir.open('foo')
        self.assertEqual(gd.control_url.rstrip('/'), urlutils.local_path_to_url(os.path.abspath(gitrepo.controldir())))

    def test_shared_repository(self):
        t = get_transport('.')
        self.assertRaises(errors.SharedRepositoriesUnsupported, dir.LocalGitControlDirFormat().initialize_on_transport_ex, t, shared_repo=True)