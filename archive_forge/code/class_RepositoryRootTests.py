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
class RepositoryRootTests(TestCase):

    def mkdtemp(self):
        return tempfile.mkdtemp()

    def open_repo(self, name):
        temp_dir = self.mkdtemp()
        repo = open_repo(name, temp_dir)
        self.addCleanup(tear_down_repo, repo)
        return repo

    def test_simple_props(self):
        r = self.open_repo('a.git')
        self.assertEqual(r.controldir(), r.path)

    def test_setitem(self):
        r = self.open_repo('a.git')
        r[b'refs/tags/foo'] = b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'
        self.assertEqual(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', r[b'refs/tags/foo'].id)

    def test_getitem_unicode(self):
        r = self.open_repo('a.git')
        test_keys = [(b'refs/heads/master', True), (b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', True), (b'11' * 19 + b'--', False)]
        for k, contained in test_keys:
            self.assertEqual(k in r, contained)
        if getattr(self, 'assertRaisesRegex', None):
            assertRaisesRegexp = self.assertRaisesRegex
        else:
            assertRaisesRegexp = self.assertRaisesRegexp
        for k, _ in test_keys:
            assertRaisesRegexp(TypeError, "'name' must be bytestring, not int", r.__getitem__, 12)

    def test_delitem(self):
        r = self.open_repo('a.git')
        del r[b'refs/heads/master']
        self.assertRaises(KeyError, lambda: r[b'refs/heads/master'])
        del r[b'HEAD']
        self.assertRaises(KeyError, lambda: r[b'HEAD'])
        self.assertRaises(ValueError, r.__delitem__, b'notrefs/foo')

    def test_get_refs(self):
        r = self.open_repo('a.git')
        self.assertEqual({b'HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/heads/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, r.get_refs())

    def test_head(self):
        r = self.open_repo('a.git')
        self.assertEqual(r.head(), b'a90fa2d900a17e99b433217e988c4eb4a2e9a097')

    def test_get_object(self):
        r = self.open_repo('a.git')
        obj = r.get_object(r.head())
        self.assertEqual(obj.type_name, b'commit')

    def test_get_object_non_existant(self):
        r = self.open_repo('a.git')
        self.assertRaises(KeyError, r.get_object, missing_sha)

    def test_contains_object(self):
        r = self.open_repo('a.git')
        self.assertIn(r.head(), r)
        self.assertNotIn(b'z' * 40, r)

    def test_contains_ref(self):
        r = self.open_repo('a.git')
        self.assertIn(b'HEAD', r)

    def test_get_no_description(self):
        r = self.open_repo('a.git')
        self.assertIs(None, r.get_description())

    def test_get_description(self):
        r = self.open_repo('a.git')
        with open(os.path.join(r.path, 'description'), 'wb') as f:
            f.write(b'Some description')
        self.assertEqual(b'Some description', r.get_description())

    def test_set_description(self):
        r = self.open_repo('a.git')
        description = b'Some description'
        r.set_description(description)
        self.assertEqual(description, r.get_description())

    def test_contains_missing(self):
        r = self.open_repo('a.git')
        self.assertNotIn(b'bar', r)

    def test_get_peeled(self):
        r = self.open_repo('a.git')
        tag_sha = b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a'
        self.assertNotEqual(r[tag_sha].sha().hexdigest(), r.head())
        self.assertEqual(r.get_peeled(b'refs/tags/mytag'), r.head())
        packed_tag_sha = b'b0931cadc54336e78a1d980420e3268903b57a50'
        parent_sha = r[r.head()].parents[0]
        self.assertNotEqual(r[packed_tag_sha].sha().hexdigest(), parent_sha)
        self.assertEqual(r.get_peeled(b'refs/tags/mytag-packed'), parent_sha)

    def test_get_peeled_not_tag(self):
        r = self.open_repo('a.git')
        self.assertEqual(r.get_peeled(b'HEAD'), r.head())

    def test_get_parents(self):
        r = self.open_repo('a.git')
        self.assertEqual([b'2a72d929692c41d8554c07f6301757ba18a65d91'], r.get_parents(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'))
        r.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], None)
        self.assertEqual([], r.get_parents(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'))

    def test_get_walker(self):
        r = self.open_repo('a.git')
        self.assertEqual([e.commit.id for e in r.get_walker()], [r.head(), b'2a72d929692c41d8554c07f6301757ba18a65d91'])
        self.assertEqual([e.commit.id for e in r.get_walker([b'2a72d929692c41d8554c07f6301757ba18a65d91'])], [b'2a72d929692c41d8554c07f6301757ba18a65d91'])
        self.assertEqual([e.commit.id for e in r.get_walker(b'2a72d929692c41d8554c07f6301757ba18a65d91')], [b'2a72d929692c41d8554c07f6301757ba18a65d91'])

    def assertFilesystemHidden(self, path):
        if sys.platform != 'win32':
            return
        import ctypes
        from ctypes.wintypes import DWORD, LPCWSTR
        GetFileAttributesW = ctypes.WINFUNCTYPE(DWORD, LPCWSTR)(('GetFileAttributesW', ctypes.windll.kernel32))
        self.assertTrue(2 & GetFileAttributesW(path))

    def test_init_existing(self):
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        t = Repo.init(tmp_dir)
        self.addCleanup(t.close)
        self.assertEqual(os.listdir(tmp_dir), ['.git'])
        self.assertFilesystemHidden(os.path.join(tmp_dir, '.git'))

    def test_init_mkdir(self):
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo_dir = os.path.join(tmp_dir, 'a-repo')
        t = Repo.init(repo_dir, mkdir=True)
        self.addCleanup(t.close)
        self.assertEqual(os.listdir(repo_dir), ['.git'])
        self.assertFilesystemHidden(os.path.join(repo_dir, '.git'))

    def test_init_mkdir_unicode(self):
        repo_name = '§'
        try:
            os.fsencode(repo_name)
        except UnicodeEncodeError:
            self.skipTest('filesystem lacks unicode support')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo_dir = os.path.join(tmp_dir, repo_name)
        t = Repo.init(repo_dir, mkdir=True)
        self.addCleanup(t.close)
        self.assertEqual(os.listdir(repo_dir), ['.git'])
        self.assertFilesystemHidden(os.path.join(repo_dir, '.git'))

    @skipIf(sys.platform == 'win32', 'fails on Windows')
    def test_fetch(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        t = Repo.init(tmp_dir)
        self.addCleanup(t.close)
        r.fetch(t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', t)
        self.assertIn(b'b0931cadc54336e78a1d980420e3268903b57a50', t)

    @skipIf(sys.platform == 'win32', 'fails on Windows')
    def test_fetch_ignores_missing_refs(self):
        r = self.open_repo('a.git')
        missing = b'1234566789123456789123567891234657373833'
        r.refs[b'refs/heads/blah'] = missing
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        t = Repo.init(tmp_dir)
        self.addCleanup(t.close)
        r.fetch(t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', t)
        self.assertIn(b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', t)
        self.assertIn(b'b0931cadc54336e78a1d980420e3268903b57a50', t)
        self.assertNotIn(missing, t)

    def test_clone(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        with r.clone(tmp_dir, mkdir=False) as t:
            self.assertEqual({b'HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/remotes/origin/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/remotes/origin/HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/heads/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, t.refs.as_dict())
            shas = [e.commit.id for e in r.get_walker()]
            self.assertEqual(shas, [t.head(), b'2a72d929692c41d8554c07f6301757ba18a65d91'])
            c = t.get_config()
            encoded_path = r.path
            if not isinstance(encoded_path, bytes):
                encoded_path = os.fsencode(encoded_path)
            self.assertEqual(encoded_path, c.get((b'remote', b'origin'), b'url'))
            self.assertEqual(b'+refs/heads/*:refs/remotes/origin/*', c.get((b'remote', b'origin'), b'fetch'))

    def test_clone_no_head(self):
        temp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        repo_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'testdata', 'repos')
        dest_dir = os.path.join(temp_dir, 'a.git')
        shutil.copytree(os.path.join(repo_dir, 'a.git'), dest_dir, symlinks=True)
        r = Repo(dest_dir)
        self.addCleanup(r.close)
        del r.refs[b'refs/heads/master']
        del r.refs[b'HEAD']
        t = r.clone(os.path.join(temp_dir, 'b.git'), mkdir=True)
        self.addCleanup(t.close)
        self.assertEqual({b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, t.refs.as_dict())

    def test_clone_empty(self):
        """Test clone() doesn't crash if HEAD points to a non-existing ref.

        This simulates cloning server-side bare repository either when it is
        still empty or if user renames master branch and pushes private repo
        to the server.
        Non-bare repo HEAD always points to an existing ref.
        """
        r = self.open_repo('empty.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        r.clone(tmp_dir, mkdir=False, bare=True)

    def test_reset_index_symlink_enabled(self):
        if sys.platform == 'win32':
            self.skipTest('symlinks are not supported on Windows')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        o = Repo.init(os.path.join(tmp_dir, 's'), mkdir=True)
        os.symlink('foo', os.path.join(tmp_dir, 's', 'bar'))
        o.stage('bar')
        o.do_commit(b'add symlink')
        t = o.clone(os.path.join(tmp_dir, 't'), symlinks=True)
        o.close()
        bar_path = os.path.join(tmp_dir, 't', 'bar')
        if sys.platform == 'win32':
            with open(bar_path) as f:
                self.assertEqual('foo', f.read())
        else:
            self.assertEqual('foo', os.readlink(bar_path))
        t.close()

    def test_reset_index_symlink_disabled(self):
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        o = Repo.init(os.path.join(tmp_dir, 's'), mkdir=True)
        o.close()
        os.symlink('foo', os.path.join(tmp_dir, 's', 'bar'))
        o.stage('bar')
        o.do_commit(b'add symlink')
        t = o.clone(os.path.join(tmp_dir, 't'), symlinks=False)
        with open(os.path.join(tmp_dir, 't', 'bar')) as f:
            self.assertEqual('foo', f.read())
        t.close()

    def test_clone_bare(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        t = r.clone(tmp_dir, mkdir=False)
        t.close()

    def test_clone_checkout_and_bare(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        self.assertRaises(ValueError, r.clone, tmp_dir, mkdir=False, checkout=True, bare=True)

    def test_clone_branch(self):
        r = self.open_repo('a.git')
        r.refs[b'refs/heads/mybranch'] = b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a'
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        with r.clone(tmp_dir, mkdir=False, branch=b'mybranch') as t:
            chain, sha = t.refs.follow(b'HEAD')
            self.assertEqual(chain[-1], b'refs/heads/mybranch')
            self.assertEqual(sha, b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a')
            self.assertEqual(t.refs[b'refs/remotes/origin/HEAD'], b'a90fa2d900a17e99b433217e988c4eb4a2e9a097')

    def test_clone_tag(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        with r.clone(tmp_dir, mkdir=False, branch=b'mytag') as t:
            self.assertEqual(t.refs.read_ref(b'HEAD'), b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a')
            self.assertEqual(t.refs[b'refs/remotes/origin/HEAD'], b'a90fa2d900a17e99b433217e988c4eb4a2e9a097')

    def test_clone_invalid_branch(self):
        r = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        self.assertRaises(ValueError, r.clone, tmp_dir, mkdir=False, branch=b'mybranch')

    def test_merge_history(self):
        r = self.open_repo('simple_merge.git')
        shas = [e.commit.id for e in r.get_walker()]
        self.assertEqual(shas, [b'5dac377bdded4c9aeb8dff595f0faeebcc8498cc', b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310'])

    def test_out_of_order_merge(self):
        """Test that revision history is ordered by date, not parent order."""
        r = self.open_repo('ooo_merge.git')
        shas = [e.commit.id for e in r.get_walker()]
        self.assertEqual(shas, [b'7601d7f6231db6a57f7bbb79ee52e4d462fd44d1', b'f507291b64138b875c28e03469025b1ea20bc614', b'fb5b0425c7ce46959bec94d54b9a157645e114f5', b'f9e39b120c68182a4ba35349f832d0e4e61f485c'])

    def test_get_tags_empty(self):
        r = self.open_repo('ooo_merge.git')
        self.assertEqual({}, r.refs.as_dict(b'refs/tags'))

    def test_get_config(self):
        r = self.open_repo('ooo_merge.git')
        self.assertIsInstance(r.get_config(), Config)

    def test_get_config_stack(self):
        r = self.open_repo('ooo_merge.git')
        self.assertIsInstance(r.get_config_stack(), Config)

    def test_common_revisions(self):
        """This test demonstrates that ``find_common_revisions()`` actually
        returns common heads, not revisions; dulwich already uses
        ``find_common_revisions()`` in such a manner (see
        ``Repo.find_objects()``).
        """
        expected_shas = {b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e'}
        r_base = self.open_repo('simple_merge.git')
        r1_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, r1_dir)
        r1_commits = [b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310']
        r2_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, r2_dir)
        r2_commits = [b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6', b'60dacdc733de308bb77bb76ce0fb0f9b44c9769e', b'0d89f20333fbb1d2f3a94da77f4981373d8f4310']
        r1 = Repo.init_bare(r1_dir)
        for c in r1_commits:
            r1.object_store.add_object(r_base.get_object(c))
        r1.refs[b'HEAD'] = r1_commits[0]
        r2 = Repo.init_bare(r2_dir)
        for c in r2_commits:
            r2.object_store.add_object(r_base.get_object(c))
        r2.refs[b'HEAD'] = r2_commits[0]
        shas = r2.object_store.find_common_revisions(r1.get_graph_walker())
        self.assertEqual(set(shas), expected_shas)
        shas = r1.object_store.find_common_revisions(r2.get_graph_walker())
        self.assertEqual(set(shas), expected_shas)

    def test_shell_hook_pre_commit(self):
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        pre_commit_fail = '#!/bin/sh\nexit 1\n'
        pre_commit_success = '#!/bin/sh\nexit 0\n'
        repo_dir = os.path.join(self.mkdtemp())
        self.addCleanup(shutil.rmtree, repo_dir)
        r = Repo.init(repo_dir)
        self.addCleanup(r.close)
        pre_commit = os.path.join(r.controldir(), 'hooks', 'pre-commit')
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_fail)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        self.assertRaises(errors.CommitError, r.do_commit, b'failed commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_success)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        commit_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([], r[commit_sha].parents)

    def test_shell_hook_commit_msg(self):
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        commit_msg_fail = '#!/bin/sh\nexit 1\n'
        commit_msg_success = '#!/bin/sh\nexit 0\n'
        repo_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        r = Repo.init(repo_dir)
        self.addCleanup(r.close)
        commit_msg = os.path.join(r.controldir(), 'hooks', 'commit-msg')
        with open(commit_msg, 'w') as f:
            f.write(commit_msg_fail)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        self.assertRaises(errors.CommitError, r.do_commit, b'failed commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        with open(commit_msg, 'w') as f:
            f.write(commit_msg_success)
        os.chmod(commit_msg, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        commit_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([], r[commit_sha].parents)

    def test_shell_hook_pre_commit_add_files(self):
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        pre_commit_contents = "#!{executable}\nimport sys\nsys.path.extend({path!r})\nfrom dulwich.repo import Repo\n\nwith open('foo', 'w') as f:\n    f.write('newfile')\n\nr = Repo('.')\nr.stage(['foo'])\n".format(executable=sys.executable, path=[os.path.join(os.path.dirname(__file__), '..', '..'), *sys.path])
        repo_dir = os.path.join(self.mkdtemp())
        self.addCleanup(shutil.rmtree, repo_dir)
        r = Repo.init(repo_dir)
        self.addCleanup(r.close)
        with open(os.path.join(repo_dir, 'blah'), 'w') as f:
            f.write('blah')
        r.stage(['blah'])
        pre_commit = os.path.join(r.controldir(), 'hooks', 'pre-commit')
        with open(pre_commit, 'w') as f:
            f.write(pre_commit_contents)
        os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        commit_sha = r.do_commit(b'new commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
        self.assertEqual([], r[commit_sha].parents)
        tree = r[r[commit_sha].tree]
        self.assertEqual({b'blah', b'foo'}, set(tree))

    def test_shell_hook_post_commit(self):
        if os.name != 'posix':
            self.skipTest('shell hook tests requires POSIX shell')
        repo_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, repo_dir)
        r = Repo.init(repo_dir)
        self.addCleanup(r.close)
        fd, path = tempfile.mkstemp(dir=repo_dir)
        os.close(fd)
        post_commit_msg = '#!/bin/sh\nrm ' + path + '\n'
        root_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        self.assertEqual([], r[root_sha].parents)
        post_commit = os.path.join(r.controldir(), 'hooks', 'post-commit')
        with open(post_commit, 'wb') as f:
            f.write(post_commit_msg.encode(locale.getpreferredencoding()))
        os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        commit_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        self.assertEqual([root_sha], r[commit_sha].parents)
        self.assertFalse(os.path.exists(path))
        post_commit_msg_fail = '#!/bin/sh\nexit 1\n'
        with open(post_commit, 'w') as f:
            f.write(post_commit_msg_fail)
        os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        warnings.simplefilter('always', UserWarning)
        self.addCleanup(warnings.resetwarnings)
        warnings_list, restore_warnings = setup_warning_catcher()
        self.addCleanup(restore_warnings)
        commit_sha2 = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        expected_warning = UserWarning('post-commit hook failed: Hook post-commit exited with non-zero status 1')
        for w in warnings_list:
            if type(w) is type(expected_warning) and w.args == expected_warning.args:
                break
        else:
            raise AssertionError(f'Expected warning {expected_warning!r} not in {warnings_list!r}')
        self.assertEqual([commit_sha], r[commit_sha2].parents)

    def test_as_dict(self):

        def check(repo):
            self.assertEqual(repo.refs.subkeys(b'refs/tags'), repo.refs.subkeys(b'refs/tags/'))
            self.assertEqual(repo.refs.as_dict(b'refs/tags'), repo.refs.as_dict(b'refs/tags/'))
            self.assertEqual(repo.refs.as_dict(b'refs/heads'), repo.refs.as_dict(b'refs/heads/'))
        bare = self.open_repo('a.git')
        tmp_dir = self.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        with bare.clone(tmp_dir, mkdir=False) as nonbare:
            check(nonbare)
            check(bare)

    def test_working_tree(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        worktree_temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, worktree_temp_dir)
        r = Repo.init(temp_dir)
        self.addCleanup(r.close)
        root_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        r.refs[b'refs/heads/master'] = root_sha
        w = Repo._init_new_working_directory(worktree_temp_dir, r)
        self.addCleanup(w.close)
        new_sha = w.do_commit(b'new commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
        w.refs[b'HEAD'] = new_sha
        self.assertEqual(os.path.abspath(r.controldir()), os.path.abspath(w.commondir()))
        self.assertEqual(r.refs.keys(), w.refs.keys())
        self.assertNotEqual(r.head(), w.head())