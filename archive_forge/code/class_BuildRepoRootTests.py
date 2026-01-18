import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
class BuildRepoRootTests(TestCase):
    """Tests that build on-disk repos from scratch.

    Repos live in a temp dir and are torn down after each test. They start with
    a single commit in master having single file named 'a'.
    """

    def get_repo_dir(self):
        return os.path.join(tempfile.mkdtemp(), 'test')

    def setUp(self):
        super().setUp()
        self._repo_dir = self.get_repo_dir()
        os.makedirs(self._repo_dir)
        r = self._repo = Repo.init(self._repo_dir)
        self.addCleanup(tear_down_repo, r)
        self.assertFalse(r.bare)
        self.assertEqual(b'ref: refs/heads/master', r.refs.read_ref(b'HEAD'))
        self.assertRaises(KeyError, lambda: r.refs[b'refs/heads/master'])
        with open(os.path.join(r.path, 'a'), 'wb') as f:
            f.write(b'file contents')
        r.stage(['a'])
        commit_sha = r.do_commit(b'msg', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        self.assertEqual([], r[commit_sha].parents)
        self._root_commit = commit_sha

    def test_get_shallow(self):
        self.assertEqual(set(), self._repo.get_shallow())
        with open(os.path.join(self._repo.path, '.git', 'shallow'), 'wb') as f:
            f.write(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097\n')
        self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())

    def test_update_shallow(self):
        self._repo.update_shallow(None, None)
        self.assertEqual(set(), self._repo.get_shallow())
        self._repo.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], None)
        self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())
        self._repo.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], [b'f9e39b120c68182a4ba35349f832d0e4e61f485c'])
        self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())
        self._repo.update_shallow(None, [b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'])
        self.assertEqual(set(), self._repo.get_shallow())
        self.assertEqual(False, os.path.exists(os.path.join(self._repo.controldir(), 'shallow')))

    def test_build_repo(self):
        r = self._repo
        self.assertEqual(b'ref: refs/heads/master', r.refs.read_ref(b'HEAD'))
        self.assertEqual(self._root_commit, r.refs[b'refs/heads/master'])
        expected_blob = objects.Blob.from_string(b'file contents')
        self.assertEqual(expected_blob.data, r[expected_blob.id].data)
        actual_commit = r[self._root_commit]
        self.assertEqual(b'msg', actual_commit.message)

    def test_commit_modified(self):
        r = self._repo
        with open(os.path.join(r.path, 'a'), 'wb') as f:
            f.write(b'new contents')
        r.stage(['a'])
        commit_sha = r.do_commit(b'modified a', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([self._root_commit], r[commit_sha].parents)
        a_mode, a_id = tree_lookup_path(r.get_object, r[commit_sha].tree, b'a')
        self.assertEqual(stat.S_IFREG | 420, a_mode)
        self.assertEqual(b'new contents', r[a_id].data)

    @skipIf(not getattr(os, 'symlink', None), 'Requires symlink support')
    def test_commit_symlink(self):
        r = self._repo
        os.symlink('a', os.path.join(r.path, 'b'))
        r.stage(['a', 'b'])
        commit_sha = r.do_commit(b'Symlink b', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([self._root_commit], r[commit_sha].parents)
        b_mode, b_id = tree_lookup_path(r.get_object, r[commit_sha].tree, b'b')
        self.assertTrue(stat.S_ISLNK(b_mode))
        self.assertEqual(b'a', r[b_id].data)

    def test_commit_merge_heads_file(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        r = Repo.init(tmp_dir)
        with open(os.path.join(r.path, 'a'), 'w') as f:
            f.write('initial text')
        c1 = r.do_commit(b'initial commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        with open(os.path.join(r.path, 'a'), 'w') as f:
            f.write('merged text')
        with open(os.path.join(r.path, '.git', 'MERGE_HEAD'), 'w') as f:
            f.write('c27a2d21dd136312d7fa9e8baabb82561a1727d0\n')
        r.stage(['a'])
        commit_sha = r.do_commit(b'deleted a', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([c1, b'c27a2d21dd136312d7fa9e8baabb82561a1727d0'], r[commit_sha].parents)

    def test_commit_deleted(self):
        r = self._repo
        os.remove(os.path.join(r.path, 'a'))
        r.stage(['a'])
        commit_sha = r.do_commit(b'deleted a', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([self._root_commit], r[commit_sha].parents)
        self.assertEqual([], list(r.open_index()))
        tree = r[r[commit_sha].tree]
        self.assertEqual([], list(tree.iteritems()))

    def test_commit_follows(self):
        r = self._repo
        r.refs.set_symbolic_ref(b'HEAD', b'refs/heads/bla')
        commit_sha = r.do_commit(b'commit with strange character', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'HEAD')
        self.assertEqual(commit_sha, r[b'refs/heads/bla'].id)

    def test_commit_encoding(self):
        r = self._repo
        commit_sha = r.do_commit(b'commit with strange character \xee', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, encoding=b'iso8859-1')
        self.assertEqual(b'iso8859-1', r[commit_sha].encoding)

    def test_compression_level(self):
        r = self._repo
        c = r.get_config()
        c.set(('core',), 'compression', '3')
        c.set(('core',), 'looseCompression', '4')
        c.write_to_path()
        r = Repo(self._repo_dir)
        self.assertEqual(r.object_store.loose_compression_level, 4)

    def test_repositoryformatversion_unsupported(self):
        r = self._repo
        c = r.get_config()
        c.set(('core',), 'repositoryformatversion', '2')
        c.write_to_path()
        self.assertRaises(UnsupportedVersion, Repo, self._repo_dir)

    def test_repositoryformatversion_1(self):
        r = self._repo
        c = r.get_config()
        c.set(('core',), 'repositoryformatversion', '1')
        c.write_to_path()
        Repo(self._repo_dir)

    def test_worktreeconfig_extension(self):
        r = self._repo
        c = r.get_config()
        c.set(('core',), 'repositoryformatversion', '1')
        c.set(('extensions',), 'worktreeconfig', True)
        c.write_to_path()
        c = r.get_worktree_config()
        c.set(('user',), 'repositoryformatversion', '1')
        c.set((b'user',), b'name', b'Jelmer')
        c.write_to_path()
        cs = r.get_config_stack()
        self.assertEqual(cs.get(('user',), 'name'), b'Jelmer')

    def test_repositoryformatversion_1_extension(self):
        r = self._repo
        c = r.get_config()
        c.set(('core',), 'repositoryformatversion', '1')
        c.set(('extensions',), 'unknownextension', True)
        c.write_to_path()
        self.assertRaises(UnsupportedExtension, Repo, self._repo_dir)

    def test_commit_encoding_from_config(self):
        r = self._repo
        c = r.get_config()
        c.set(('i18n',), 'commitEncoding', 'iso8859-1')
        c.write_to_path()
        commit_sha = r.do_commit(b'commit with strange character \xee', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual(b'iso8859-1', r[commit_sha].encoding)

    def test_commit_config_identity(self):
        r = self._repo
        c = r.get_config()
        c.set((b'user',), b'name', b'Jelmer')
        c.set((b'user',), b'email', b'jelmer@apache.org')
        c.write_to_path()
        commit_sha = r.do_commit(b'message')
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].author)
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].committer)

    def test_commit_config_identity_strips_than(self):
        r = self._repo
        c = r.get_config()
        c.set((b'user',), b'name', b'Jelmer')
        c.set((b'user',), b'email', b'<jelmer@apache.org>')
        c.write_to_path()
        commit_sha = r.do_commit(b'message')
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].author)
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].committer)

    def test_commit_config_identity_in_memoryrepo(self):
        r = MemoryRepo.init_bare([], {})
        c = r.get_config()
        c.set((b'user',), b'name', b'Jelmer')
        c.set((b'user',), b'email', b'jelmer@apache.org')
        commit_sha = r.do_commit(b'message', tree=objects.Tree().id)
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].author)
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].committer)

    def test_commit_config_identity_from_env(self):
        self.overrideEnv('GIT_COMMITTER_NAME', 'joe')
        self.overrideEnv('GIT_COMMITTER_EMAIL', 'joe@example.com')
        r = self._repo
        c = r.get_config()
        c.set((b'user',), b'name', b'Jelmer')
        c.set((b'user',), b'email', b'jelmer@apache.org')
        c.write_to_path()
        commit_sha = r.do_commit(b'message')
        self.assertEqual(b'Jelmer <jelmer@apache.org>', r[commit_sha].author)
        self.assertEqual(b'joe <joe@example.com>', r[commit_sha].committer)

    def test_commit_fail_ref(self):
        r = self._repo

        def set_if_equals(name, old_ref, new_ref, **kwargs):
            return False
        r.refs.set_if_equals = set_if_equals

        def add_if_new(name, new_ref, **kwargs):
            self.fail('Unexpected call to add_if_new')
        r.refs.add_if_new = add_if_new
        old_shas = set(r.object_store)
        self.assertRaises(errors.CommitError, r.do_commit, b'failed commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        new_shas = set(r.object_store) - old_shas
        self.assertEqual(1, len(new_shas))
        new_commit = r[new_shas.pop()]
        self.assertEqual(r[self._root_commit].tree, new_commit.tree)
        self.assertEqual(b'failed commit', new_commit.message)

    def test_commit_branch(self):
        r = self._repo
        commit_sha = r.do_commit(b'commit to branch', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'refs/heads/new_branch')
        self.assertEqual(self._root_commit, r[b'HEAD'].id)
        self.assertEqual(commit_sha, r[b'refs/heads/new_branch'].id)
        self.assertEqual([], r[commit_sha].parents)
        self.assertIn(b'refs/heads/new_branch', r)
        new_branch_head = commit_sha
        commit_sha = r.do_commit(b'commit to branch 2', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'refs/heads/new_branch')
        self.assertEqual(self._root_commit, r[b'HEAD'].id)
        self.assertEqual(commit_sha, r[b'refs/heads/new_branch'].id)
        self.assertEqual([new_branch_head], r[commit_sha].parents)

    def test_commit_merge_heads(self):
        r = self._repo
        merge_1 = r.do_commit(b'commit to branch 2', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=b'refs/heads/new_branch')
        commit_sha = r.do_commit(b'commit with merge', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, merge_heads=[merge_1])
        self.assertEqual([self._root_commit, merge_1], r[commit_sha].parents)

    def test_commit_dangling_commit(self):
        r = self._repo
        old_shas = set(r.object_store)
        old_refs = r.get_refs()
        commit_sha = r.do_commit(b'commit with no ref', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=None)
        new_shas = set(r.object_store) - old_shas
        self.assertEqual(1, len(new_shas))
        new_commit = r[new_shas.pop()]
        self.assertEqual(r[self._root_commit].tree, new_commit.tree)
        self.assertEqual([], r[commit_sha].parents)
        self.assertEqual(old_refs, r.get_refs())

    def test_commit_dangling_commit_with_parents(self):
        r = self._repo
        old_shas = set(r.object_store)
        old_refs = r.get_refs()
        commit_sha = r.do_commit(b'commit with no ref', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=None, merge_heads=[self._root_commit])
        new_shas = set(r.object_store) - old_shas
        self.assertEqual(1, len(new_shas))
        new_commit = r[new_shas.pop()]
        self.assertEqual(r[self._root_commit].tree, new_commit.tree)
        self.assertEqual([self._root_commit], r[commit_sha].parents)
        self.assertEqual(old_refs, r.get_refs())

    def test_stage_absolute(self):
        r = self._repo
        os.remove(os.path.join(r.path, 'a'))
        self.assertRaises(ValueError, r.stage, [os.path.join(r.path, 'a')])

    def test_stage_deleted(self):
        r = self._repo
        os.remove(os.path.join(r.path, 'a'))
        r.stage(['a'])
        r.stage(['a'])
        self.assertEqual([], list(r.open_index()))

    def test_stage_directory(self):
        r = self._repo
        os.mkdir(os.path.join(r.path, 'c'))
        r.stage(['c'])
        self.assertEqual([b'a'], list(r.open_index()))

    def test_stage_submodule(self):
        r = self._repo
        s = Repo.init(os.path.join(r.path, 'sub'), mkdir=True)
        s.do_commit(b'message')
        r.stage(['sub'])
        self.assertEqual([b'a', b'sub'], list(r.open_index()))

    def test_unstage_midify_file_with_dir(self):
        os.mkdir(os.path.join(self._repo.path, 'new_dir'))
        full_path = os.path.join(self._repo.path, 'new_dir', 'foo')
        with open(full_path, 'w') as f:
            f.write('hello')
        porcelain.add(self._repo, paths=[full_path])
        porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        with open(full_path, 'a') as f:
            f.write('something new')
        self._repo.unstage(['new_dir/foo'])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'new_dir/foo'], []], status)

    def test_unstage_while_no_commit(self):
        file = 'foo'
        full_path = os.path.join(self._repo.path, file)
        with open(full_path, 'w') as f:
            f.write('hello')
        porcelain.add(self._repo, paths=[full_path])
        self._repo.unstage([file])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['foo']], status)

    def test_unstage_add_file(self):
        file = 'foo'
        full_path = os.path.join(self._repo.path, file)
        porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        with open(full_path, 'w') as f:
            f.write('hello')
        porcelain.add(self._repo, paths=[full_path])
        self._repo.unstage([file])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['foo']], status)

    def test_unstage_modify_file(self):
        file = 'foo'
        full_path = os.path.join(self._repo.path, file)
        with open(full_path, 'w') as f:
            f.write('hello')
        porcelain.add(self._repo, paths=[full_path])
        porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        with open(full_path, 'a') as f:
            f.write('broken')
        porcelain.add(self._repo, paths=[full_path])
        self._repo.unstage([file])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)

    def test_unstage_remove_file(self):
        file = 'foo'
        full_path = os.path.join(self._repo.path, file)
        with open(full_path, 'w') as f:
            f.write('hello')
        porcelain.add(self._repo, paths=[full_path])
        porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        os.remove(full_path)
        self._repo.unstage([file])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)

    def test_reset_index(self):
        r = self._repo
        with open(os.path.join(r.path, 'a'), 'wb') as f:
            f.write(b'changed')
        with open(os.path.join(r.path, 'b'), 'wb') as f:
            f.write(b'added')
        r.stage(['a', 'b'])
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [b'b'], 'delete': [], 'modify': [b'a']}, [], []], status)
        r.reset_index()
        status = list(porcelain.status(self._repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['b']], status)

    @skipIf(sys.platform in ('win32', 'darwin'), 'tries to implicitly decode as utf8')
    def test_commit_no_encode_decode(self):
        r = self._repo
        repo_path_bytes = os.fsencode(r.path)
        encodings = ('utf8', 'latin1')
        names = ['À'.encode(encoding) for encoding in encodings]
        for name, encoding in zip(names, encodings):
            full_path = os.path.join(repo_path_bytes, name)
            with open(full_path, 'wb') as f:
                f.write(encoding.encode('ascii'))
            self.addCleanup(os.remove, full_path)
        r.stage(names)
        commit_sha = r.do_commit(b'Files with different encodings', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0, ref=None, merge_heads=[self._root_commit])
        for name, encoding in zip(names, encodings):
            mode, id = tree_lookup_path(r.get_object, r[commit_sha].tree, name)
            self.assertEqual(stat.S_IFREG | 420, mode)
            self.assertEqual(encoding.encode('ascii'), r[id].data)

    def test_discover_intended(self):
        path = os.path.join(self._repo_dir, 'b/c')
        r = Repo.discover(path)
        self.assertEqual(r.head(), self._repo.head())

    def test_discover_isrepo(self):
        r = Repo.discover(self._repo_dir)
        self.assertEqual(r.head(), self._repo.head())

    def test_discover_notrepo(self):
        with self.assertRaises(NotGitRepository):
            Repo.discover('/')