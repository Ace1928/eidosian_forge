import os
from dulwich.repo import Repo as GitRepo
from ...controldir import ControlDir
from ...tests.blackbox import ExternalBase
from ...tests.features import PluginLoadedFeature
from ...tests.script import TestCaseWithTransportAndScript
from ...workingtree import WorkingTree
from .. import tests
class TestGitBlackBox(ExternalBase):

    def simple_commit(self):
        repo = GitRepo.init(self.test_dir)
        builder = tests.GitBranchBuilder()
        builder.set_file('a', b'text for a\n', False)
        r1 = builder.commit(b'Joe Foo <joe@foo.com>', '<The commit message>')
        return (repo, builder.finish()[r1])

    def test_add(self):
        r = GitRepo.init(self.test_dir)
        dir = ControlDir.open(self.test_dir)
        dir.create_branch()
        self.build_tree(['a', 'b'])
        output, error = self.run_bzr(['add', 'a'])
        self.assertEqual('adding a\n', output)
        self.assertEqual('', error)
        output, error = self.run_bzr(['add', '--file-ids-from=../othertree', 'b'])
        self.assertEqual('adding b\n', output)
        self.assertEqual('Ignoring --file-ids-from, since the tree does not support setting file ids.\n', error)

    def test_nick(self):
        r = GitRepo.init(self.test_dir)
        dir = ControlDir.open(self.test_dir)
        dir.create_branch()
        output, error = self.run_bzr(['nick'])
        self.assertEqual('master\n', output)

    def test_branches(self):
        self.simple_commit()
        output, error = self.run_bzr(['branches'])
        self.assertEqual('* master\n', output)

    def test_info(self):
        self.simple_commit()
        output, error = self.run_bzr(['info'])
        self.assertEqual(error, '')
        self.assertEqual(output, 'Standalone tree (format: git)\nLocation:\n            light checkout root: .\n  checkout of co-located branch: master\n')

    def test_ignore(self):
        self.simple_commit()
        output, error = self.run_bzr(['ignore', 'foo'])
        self.assertEqual(error, '')
        self.assertEqual(output, '')
        self.assertFileEqual('foo\n', '.gitignore')

    def test_cat_revision(self):
        self.simple_commit()
        output, error = self.run_bzr(['cat-revision', '-r-1'], retcode=3)
        self.assertContainsRe(error, 'brz: ERROR: Repository .* does not support access to raw revision texts')
        self.assertEqual(output, '')

    def test_branch(self):
        os.mkdir('gitbranch')
        GitRepo.init(os.path.join(self.test_dir, 'gitbranch'))
        os.chdir('gitbranch')
        builder = tests.GitBranchBuilder()
        builder.set_file(b'a', b'text for a\n', False)
        builder.commit(b'Joe Foo <joe@foo.com>', b'<The commit message>')
        builder.finish()
        os.chdir('..')
        output, error = self.run_bzr(['branch', 'gitbranch', 'bzrbranch'])
        errlines = error.splitlines(False)
        self.assertTrue('Branched 1 revision(s).' in errlines or 'Branched 1 revision.' in errlines, errlines)

    def test_checkout(self):
        os.mkdir('gitbranch')
        GitRepo.init(os.path.join(self.test_dir, 'gitbranch'))
        os.chdir('gitbranch')
        builder = tests.GitBranchBuilder()
        builder.set_file(b'a', b'text for a\n', False)
        builder.commit(b'Joe Foo <joe@foo.com>', b'<The commit message>')
        builder.finish()
        os.chdir('..')
        output, error = self.run_bzr(['checkout', 'gitbranch', 'bzrbranch'])
        self.assertEqual(error, 'Fetching from Git to Bazaar repository. For better performance, fetch into a Git repository.\n')
        self.assertEqual(output, '')

    def test_branch_ls(self):
        self.simple_commit()
        output, error = self.run_bzr(['ls', '-r-1'])
        self.assertEqual(error, '')
        self.assertEqual(output, 'a\n')

    def test_init(self):
        self.run_bzr('init --format=git repo')

    def test_info_verbose(self):
        self.simple_commit()
        output, error = self.run_bzr(['info', '-v'])
        self.assertEqual(error, '')
        self.assertTrue('Standalone tree (format: git)' in output)
        self.assertTrue('control: Local Git Repository' in output)
        self.assertTrue('branch: Local Git Branch' in output)
        self.assertTrue('repository: Git Repository' in output)

    def test_push_roundtripping(self):
        self.knownFailure('roundtripping is not yet supported')
        self.with_roundtripping()
        os.mkdir('bla')
        GitRepo.init(os.path.join(self.test_dir, 'bla'))
        self.run_bzr(['init', 'foo'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        output, error = self.run_bzr(['push', '-d', 'foo', 'bla'])
        self.assertEqual(b'', output)
        self.assertTrue(error.endswith(b'Created new branch.\n'))

    def test_push_without_calculate_revnos(self):
        self.run_bzr(['init', '--git', 'bla'])
        self.run_bzr(['init', '--git', 'foo'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        output, error = self.run_bzr(['push', '-Ocalculate_revnos=no', '-d', 'foo', 'bla'])
        self.assertEqual('', output)
        self.assertContainsRe(error, 'Pushed up to revision id git(.*).\n')

    def test_merge(self):
        self.run_bzr(['init', '--git', 'orig'])
        self.build_tree_contents([('orig/a', 'orig contents\n')])
        self.run_bzr(['add', 'orig/a'])
        self.run_bzr(['commit', '-m', 'add orig', 'orig'])
        self.run_bzr(['clone', 'orig', 'other'])
        self.build_tree_contents([('other/a', 'new contents\n')])
        self.run_bzr(['commit', '-m', 'modify', 'other'])
        self.build_tree_contents([('orig/b', 'more\n')])
        self.run_bzr(['add', 'orig/b'])
        self.build_tree_contents([('orig/a', 'new contents\n')])
        self.run_bzr(['commit', '-m', 'more', 'orig'])
        self.run_bzr(['merge', '-d', 'orig', 'other'])

    def test_push_lossy_non_mainline(self):
        self.run_bzr(['init', '--git', 'bla'])
        self.run_bzr(['init', 'foo'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        self.run_bzr(['branch', 'foo', 'foo1'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo1'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        self.run_bzr(['merge', '-d', 'foo', 'foo1'])
        self.run_bzr(['commit', '--unchanged', '-m', 'merge', 'foo'])
        output, error = self.run_bzr(['push', '--lossy', '-r1.1.1', '-d', 'foo', 'bla'])
        self.assertEqual('', output)
        self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 2.\n', error)

    def test_push_lossy_non_mainline_incremental(self):
        self.run_bzr(['init', '--git', 'bla'])
        self.run_bzr(['init', 'foo'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        output, error = self.run_bzr(['push', '--lossy', '-d', 'foo', 'bla'])
        self.assertEqual('', output)
        self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 2.\n', error)
        self.run_bzr(['commit', '--unchanged', '-m', 'bla', 'foo'])
        output, error = self.run_bzr(['push', '--lossy', '-d', 'foo', 'bla'])
        self.assertEqual('', output)
        self.assertEqual('Pushing from a Bazaar to a Git repository. For better performance, push into a Bazaar repository.\nAll changes applied successfully.\nPushed up to revision 3.\n', error)

    def test_log(self):
        self.simple_commit()
        output, error = self.run_bzr(['log'])
        self.assertEqual(error, '')
        self.assertTrue('<The commit message>' in output, 'Commit message was not found in output:\n{}'.format(output))

    def test_log_verbose(self):
        self.simple_commit()
        output, error = self.run_bzr(['log', '-v'])
        self.assertContainsRe(output, 'revno: 1')

    def test_log_without_revno(self):
        self.simple_commit()
        output, error = self.run_bzr(['log', '-Ocalculate_revnos=no'])
        self.assertNotContainsRe(output, 'revno: 1')

    def test_commit_without_revno(self):
        repo = GitRepo.init(self.test_dir)
        output, error = self.run_bzr(['commit', '-Ocalculate_revnos=yes', '--unchanged', '-m', 'one'])
        self.assertContainsRe(error, 'Committed revision 1.')
        output, error = self.run_bzr(['commit', '-Ocalculate_revnos=no', '--unchanged', '-m', 'two'])
        self.assertNotContainsRe(error, 'Committed revision 2.')
        self.assertContainsRe(error, 'Committed revid .*.')

    def test_log_file(self):
        repo = GitRepo.init(self.test_dir)
        builder = tests.GitBranchBuilder()
        builder.set_file('a', b'text for a\n', False)
        r1 = builder.commit(b'Joe Foo <joe@foo.com>', 'First')
        builder.set_file('a', b'text 3a for a\n', False)
        r2a = builder.commit(b'Joe Foo <joe@foo.com>', 'Second a', base=r1)
        builder.set_file('a', b'text 3b for a\n', False)
        r2b = builder.commit(b'Joe Foo <joe@foo.com>', 'Second b', base=r1)
        builder.set_file('a', b'text 4 for a\n', False)
        builder.commit(b'Joe Foo <joe@foo.com>', 'Third', merge=[r2a], base=r2b)
        builder.finish()
        output, error = self.run_bzr(['log', '-n2', 'a'])
        self.assertEqual(error, '')
        self.assertIn('Second a', output)
        self.assertIn('Second b', output)
        self.assertIn('First', output)
        self.assertIn('Third', output)

    def test_tags(self):
        git_repo, commit_sha1 = self.simple_commit()
        git_repo.refs[b'refs/tags/foo'] = commit_sha1
        output, error = self.run_bzr(['tags'])
        self.assertEqual(error, '')
        self.assertEqual(output, 'foo                  1\n')

    def test_tag(self):
        self.simple_commit()
        output, error = self.run_bzr(['tag', 'bar'])
        self.assertEqual(error + output, 'Created tag bar.\n')

    def test_init_repo(self):
        output, error = self.run_bzr(['init', '--format=git', 'bla.git'])
        self.assertEqual(error, '')
        self.assertEqual(output, 'Created a standalone tree (format: git)\n')

    def test_diff_format(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a'])
        tree.add(['a'])
        output, error = self.run_bzr(['diff', '--format=git'], retcode=1)
        self.assertEqual(error, '')
        from dulwich import __version__ as dulwich_version
        if dulwich_version < (0, 19, 12):
            self.assertEqual(output, 'diff --git /dev/null b/a\nold mode 0\nnew mode 100644\nindex 0000000..c197bd8 100644\n--- /dev/null\n+++ b/a\n@@ -0,0 +1 @@\n+contents of a\n')
        else:
            self.assertEqual(output, 'diff --git a/a b/a\nold file mode 0\nnew file mode 100644\nindex 0000000..c197bd8 100644\n--- /dev/null\n+++ b/a\n@@ -0,0 +1 @@\n+contents of a\n')

    def test_git_import_uncolocated(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        r.do_commit(ref=b'refs/heads/bbranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        self.run_bzr(['git-import', 'a', 'b'])
        self.assertEqual({'.bzr', 'abranch', 'bbranch'}, set(os.listdir('b')))

    def test_git_import(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        r.do_commit(ref=b'refs/heads/bbranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        self.run_bzr(['git-import', '--colocated', 'a', 'b'])
        self.assertEqual({'.bzr'}, set(os.listdir('b')))
        self.assertEqual({'abranch', 'bbranch'}, set(ControlDir.open('b').branch_names()))

    def test_git_import_incremental(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        self.run_bzr(['git-import', '--colocated', 'a', 'b'])
        self.run_bzr(['git-import', '--colocated', 'a', 'b'])
        self.assertEqual({'.bzr'}, set(os.listdir('b')))
        b = ControlDir.open('b')
        self.assertEqual(['abranch'], b.branch_names())

    def test_git_import_tags(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        cid = r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        r[b'refs/tags/atag'] = cid
        self.run_bzr(['git-import', '--colocated', 'a', 'b'])
        self.assertEqual({'.bzr'}, set(os.listdir('b')))
        b = ControlDir.open('b')
        self.assertEqual(['abranch'], b.branch_names())
        self.assertEqual(['atag'], list(b.open_branch('abranch').tags.get_tag_dict().keys()))

    def test_git_import_colo(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        r.do_commit(ref=b'refs/heads/bbranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        self.make_controldir('b', format='development-colo')
        self.run_bzr(['git-import', '--colocated', 'a', 'b'])
        self.assertEqual({b.name for b in ControlDir.open('b').list_branches()}, {'abranch', 'bbranch'})

    def test_git_refs_from_git(self):
        r = GitRepo.init('a', mkdir=True)
        self.build_tree(['a/file'])
        r.stage('file')
        cid = r.do_commit(ref=b'refs/heads/abranch', committer=b'Joe <joe@example.com>', message=b'Dummy')
        r[b'refs/tags/atag'] = cid
        stdout, stderr = self.run_bzr(['git-refs', 'a'])
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, 'refs/heads/abranch -> ' + cid.decode('ascii') + '\nrefs/tags/atag -> ' + cid.decode('ascii') + '\n')

    def test_git_refs_from_bzr(self):
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/file'])
        tree.add(['file'])
        revid = tree.commit(committer=b'Joe <joe@example.com>', message=b'Dummy')
        tree.branch.tags.set_tag('atag', revid)
        stdout, stderr = self.run_bzr(['git-refs', 'a'])
        self.assertEqual(stderr, '')
        self.assertTrue('refs/tags/atag -> ' in stdout)
        self.assertTrue('HEAD -> ' in stdout)

    def test_check(self):
        r = GitRepo.init('gitr', mkdir=True)
        self.build_tree_contents([('gitr/foo', b'hello from git')])
        r.stage('foo')
        r.do_commit(b'message', committer=b'Somebody <user@example.com>')
        out, err = self.run_bzr(['check', 'gitr'])
        self.maxDiff = None
        self.assertEqual(out, '')
        self.assertTrue(err.endswith, '3 objects\n')

    def test_local_whoami(self):
        r = GitRepo.init('gitr', mkdir=True)
        self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n  name = Test User\n')])
        out, err = self.run_bzr(['whoami', '-d', 'gitr'])
        self.assertEqual(out, 'Test User <some@example.com>\n')
        self.assertEqual(err, '')
        self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n')])
        out, err = self.run_bzr(['whoami', '-d', 'gitr'])
        self.assertEqual(out, 'some@example.com\n')
        self.assertEqual(err, '')

    def test_local_signing_key(self):
        r = GitRepo.init('gitr', mkdir=True)
        self.build_tree_contents([('gitr/.git/config', '[user]\n  email = some@example.com\n  name = Test User\n  signingkey = D729A457\n')])
        out, err = self.run_bzr(['config', '-d', 'gitr', 'gpg_signing_key'])
        self.assertEqual(out, 'D729A457\n')
        self.assertEqual(err, '')