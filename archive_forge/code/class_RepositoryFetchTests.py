import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
class RepositoryFetchTests:

    def make_git_repo(self, path):
        os.mkdir(path)
        return GitRepo.init(os.path.abspath(path))

    def clone_git_repo(self, from_url, to_url, revision_id=None):
        oldrepos = self.open_git_repo(from_url)
        dir = ControlDir.create(to_url)
        newrepos = dir.create_repository()
        oldrepos.copy_content_into(newrepos, revision_id=revision_id)
        return newrepos

    def test_empty(self):
        self.make_git_repo('d')
        newrepos = self.clone_git_repo('d', 'f')
        self.assertEqual([], newrepos.all_revision_ids())

    def make_onerev_branch(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        gitsha = bb.finish()[mark]
        os.chdir('..')
        return ('d', gitsha)

    def test_single_rev(self):
        path, gitsha = self.make_onerev_branch()
        oldrepo = self.open_git_repo(path)
        newrepo = self.clone_git_repo(path, 'f')
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        self.assertEqual([revid], newrepo.all_revision_ids())

    def test_single_rev_specific(self):
        path, gitsha = self.make_onerev_branch()
        oldrepo = self.open_git_repo(path)
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        newrepo = self.clone_git_repo(path, 'f', revision_id=revid)
        self.assertEqual([revid], newrepo.all_revision_ids())

    def test_incremental(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        bb.set_file('foobar', b'fooll\nbar\n', False)
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'nextmsg')
        marks = bb.finish()
        gitsha1 = marks[mark1]
        gitsha2 = marks[mark2]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        newrepo = self.clone_git_repo('d', 'f', revision_id=revid1)
        self.assertEqual([revid1], newrepo.all_revision_ids())
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        newrepo.fetch(oldrepo, revision_id=revid2)
        self.assertEqual({revid1, revid2}, set(newrepo.all_revision_ids()))

    def test_dir_becomes_symlink(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('mylink/somefile', b'foo\nbar\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
        bb.delete_entry('mylink/somefile')
        bb.set_symlink('mylink', 'target/')
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
        marks = bb.finish()
        gitsha1 = marks[mark1]
        gitsha2 = marks[mark2]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        tree1 = newrepo.revision_tree(revid1)
        tree2 = newrepo.revision_tree(revid2)
        self.assertEqual(revid1, tree1.get_file_revision('mylink'))
        self.assertEqual('directory', tree1.kind('mylink'))
        self.assertEqual(None, tree1.get_symlink_target('mylink'))
        self.assertEqual(revid2, tree2.get_file_revision('mylink'))
        self.assertEqual('symlink', tree2.kind('mylink'))
        self.assertEqual('target/', tree2.get_symlink_target('mylink'))

    def test_symlink_becomes_dir(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_symlink('mylink', 'target/')
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
        bb.delete_entry('mylink')
        bb.set_file('mylink/somefile', b'foo\nbar\n', False)
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
        marks = bb.finish()
        gitsha1 = marks[mark1]
        gitsha2 = marks[mark2]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        tree1 = newrepo.revision_tree(revid1)
        tree2 = newrepo.revision_tree(revid2)
        self.assertEqual(revid1, tree1.get_file_revision('mylink'))
        self.assertEqual('symlink', tree1.kind('mylink'))
        self.assertEqual('target/', tree1.get_symlink_target('mylink'))
        self.assertEqual(revid2, tree2.get_file_revision('mylink'))
        self.assertEqual('directory', tree2.kind('mylink'))
        self.assertEqual(None, tree2.get_symlink_target('mylink'))

    def test_changing_symlink(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_symlink('mylink', 'target')
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
        bb.set_symlink('mylink', 'target/')
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
        marks = bb.finish()
        gitsha1 = marks[mark1]
        gitsha2 = marks[mark2]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        tree1 = newrepo.revision_tree(revid1)
        tree2 = newrepo.revision_tree(revid2)
        self.assertEqual(revid1, tree1.get_file_revision('mylink'))
        self.assertEqual('target', tree1.get_symlink_target('mylink'))
        self.assertEqual(revid2, tree2.get_file_revision('mylink'))
        self.assertEqual('target/', tree2.get_symlink_target('mylink'))

    def test_executable(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', True)
        bb.set_file('notexec', b'foo\nbar\n', False)
        mark = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        gitsha = bb.finish()[mark]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        tree = newrepo.revision_tree(revid)
        self.assertTrue(tree.has_filename('foobar'))
        self.assertEqual(True, tree.is_executable('foobar'))
        self.assertTrue(tree.has_filename('notexec'))
        self.assertEqual(False, tree.is_executable('notexec'))

    def test_becomes_executable(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'foo\nbar\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        bb.set_file('foobar', b'foo\nbar\n', True)
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        gitsha2 = bb.finish()[mark2]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        tree = newrepo.revision_tree(revid)
        self.assertTrue(tree.has_filename('foobar'))
        self.assertEqual(True, tree.is_executable('foobar'))
        self.assertEqual(revid, tree.get_file_revision('foobar'))

    def test_into_stacked_on(self):
        r = self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'foo\n', False)
        mark1 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg1')
        gitsha1 = bb.finish()[mark1]
        os.chdir('..')
        stacked_on = self.clone_git_repo('d', 'stacked-on')
        oldrepo = Repository.open('d')
        revid1 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha1)
        self.assertEqual([revid1], stacked_on.all_revision_ids())
        b = stacked_on.controldir.create_branch()
        b.generate_revision_history(revid1)
        self.assertEqual(b.last_revision(), revid1)
        tree = self.make_branch_and_tree('stacked')
        tree.branch.set_stacked_on_url(b.user_url)
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('barbar', b'bar\n', False)
        bb.set_file('foo/blie/bla', b'bla\n', False)
        mark2 = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg2')
        gitsha2 = bb.finish()[mark2]
        revid2 = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha2)
        os.chdir('..')
        tree.branch.fetch(Branch.open('d'))
        tree.branch.repository.check()
        self.addCleanup(tree.lock_read().unlock)
        self.assertEqual({(revid2,)}, tree.branch.repository.revisions.without_fallbacks().keys())
        self.assertEqual({revid1, revid2}, set(tree.branch.repository.all_revision_ids()))

    def test_non_ascii_characters(self):
        self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foőbar', b'foo\nbar\n', False)
        mark = bb.commit(b'Somebody <somebody@someorg.org>', b'mymsg')
        gitsha = bb.finish()[mark]
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        newrepo = self.clone_git_repo('d', 'f')
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        tree = newrepo.revision_tree(revid)
        self.assertTrue(tree.has_filename('foőbar'))

    def test_tagged_tree(self):
        r = self.make_git_repo('d')
        os.chdir('d')
        bb = GitBranchBuilder()
        bb.set_file('foobar', b'fooll\nbar\n', False)
        mark = bb.commit(b'Somebody <somebody@someorg.org>', b'nextmsg')
        marks = bb.finish()
        gitsha = marks[mark]
        tag = Tag()
        tag.name = b'sometag'
        tag.tag_time = int(time.time())
        tag.tag_timezone = 0
        tag.tagger = b'Somebody <somebody@example.com>'
        tag.message = b'Created tag pointed at tree'
        tag.object = (Tree, r[gitsha].tree)
        r.object_store.add_object(tag)
        r[b'refs/tags/sometag'] = tag
        os.chdir('..')
        oldrepo = self.open_git_repo('d')
        revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
        newrepo = self.clone_git_repo('d', 'f')
        self.assertEqual({revid}, set(newrepo.all_revision_ids()))