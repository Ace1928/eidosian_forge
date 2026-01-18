import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
class CheckoutTests(PorcelainTestCase):

    def setUp(self):
        super().setUp()
        self._sha, self._foo_path = _commit_file_with_content(self.repo, 'foo', 'hello\n')
        porcelain.branch_create(self.repo, 'uni')

    def test_checkout_to_existing_branch(self):
        self.assertEqual(b'master', porcelain.active_branch(self.repo))
        porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))

    def test_checkout_to_non_existing_branch(self):
        self.assertEqual(b'master', porcelain.active_branch(self.repo))
        with self.assertRaises(KeyError):
            porcelain.checkout_branch(self.repo, b'bob')
        self.assertEqual(b'master', porcelain.active_branch(self.repo))

    def test_checkout_to_branch_with_modified_files(self):
        with open(self._foo_path, 'a') as f:
            f.write('new message\n')
        porcelain.add(self.repo, paths=[self._foo_path])
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': [b'foo']}, [], []], status)
        porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': [b'foo']}, [], []], status)

    def test_checkout_with_deleted_files(self):
        porcelain.remove(self.repo.path, [os.path.join(self.repo.path, 'foo')])
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [b'foo'], 'modify': []}, [], []], status)
        porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [b'foo'], 'modify': []}, [], []], status)

    def test_checkout_to_branch_with_added_files(self):
        file_path = os.path.join(self.repo.path, 'bar')
        with open(file_path, 'w') as f:
            f.write('bar content\n')
        porcelain.add(self.repo, paths=[file_path])
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [b'bar'], 'delete': [], 'modify': []}, [], []], status)
        porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [b'bar'], 'delete': [], 'modify': []}, [], []], status)

    def test_checkout_to_branch_with_modified_file_not_present(self):
        _, nee_path = _commit_file_with_content(self.repo, 'nee', 'Good content\n')
        with open(nee_path, 'a') as f:
            f.write('bar content\n')
        porcelain.add(self.repo, paths=[nee_path])
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': [b'nee']}, [], []], status)
        with self.assertRaises(CheckoutError):
            porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'master', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': [b'nee']}, [], []], status)

    def test_checkout_to_branch_with_modified_file_not_present_forced(self):
        _, nee_path = _commit_file_with_content(self.repo, 'nee', 'Good content\n')
        with open(nee_path, 'a') as f:
            f.write('bar content\n')
        porcelain.add(self.repo, paths=[nee_path])
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': [b'nee']}, [], []], status)
        porcelain.checkout_branch(self.repo, b'uni', force=True)
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)

    def test_checkout_to_branch_with_unstaged_files(self):
        with open(self._foo_path, 'a') as f:
            f.write('new message')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)
        porcelain.checkout_branch(self.repo, b'uni')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)

    def test_checkout_to_branch_with_untracked_files(self):
        with open(os.path.join(self.repo.path, 'neu'), 'a') as f:
            f.write('new message\n')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['neu']], status)
        porcelain.checkout_branch(self.repo, b'uni')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], ['neu']], status)

    def test_checkout_to_branch_with_new_files(self):
        porcelain.checkout_branch(self.repo, b'uni')
        sub_directory = os.path.join(self.repo.path, 'sub1')
        os.mkdir(sub_directory)
        for index in range(5):
            _commit_file_with_content(self.repo, 'new_file_' + str(index + 1), 'Some content\n')
            _commit_file_with_content(self.repo, os.path.join('sub1', 'new_file_' + str(index + 10)), 'Good content\n')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        porcelain.checkout_branch(self.repo, b'master')
        self.assertEqual(b'master', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        porcelain.checkout_branch(self.repo, b'uni')
        self.assertEqual(b'uni', porcelain.active_branch(self.repo))
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)

    def test_checkout_to_branch_with_file_in_sub_directory(self):
        sub_directory = os.path.join(self.repo.path, 'sub1', 'sub2')
        os.makedirs(sub_directory)
        sub_directory_file = os.path.join(sub_directory, 'neu')
        with open(sub_directory_file, 'w') as f:
            f.write('new message\n')
        porcelain.add(self.repo, paths=[sub_directory_file])
        porcelain.commit(self.repo, message=b'add ' + sub_directory_file.encode(), committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        self.assertTrue(os.path.isdir(sub_directory))
        self.assertTrue(os.path.isdir(os.path.dirname(sub_directory)))
        porcelain.checkout_branch(self.repo, b'uni')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        self.assertFalse(os.path.isdir(sub_directory))
        self.assertFalse(os.path.isdir(os.path.dirname(sub_directory)))
        porcelain.checkout_branch(self.repo, b'master')
        self.assertTrue(os.path.isdir(sub_directory))
        self.assertTrue(os.path.isdir(os.path.dirname(sub_directory)))

    def test_checkout_to_branch_with_multiple_files_in_sub_directory(self):
        sub_directory = os.path.join(self.repo.path, 'sub1', 'sub2')
        os.makedirs(sub_directory)
        sub_directory_file_1 = os.path.join(sub_directory, 'neu')
        with open(sub_directory_file_1, 'w') as f:
            f.write('new message\n')
        sub_directory_file_2 = os.path.join(sub_directory, 'gus')
        with open(sub_directory_file_2, 'w') as f:
            f.write('alternative message\n')
        porcelain.add(self.repo, paths=[sub_directory_file_1, sub_directory_file_2])
        porcelain.commit(self.repo, message=b'add files neu and gus.', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        self.assertTrue(os.path.isdir(sub_directory))
        self.assertTrue(os.path.isdir(os.path.dirname(sub_directory)))
        porcelain.checkout_branch(self.repo, b'uni')
        status = list(porcelain.status(self.repo))
        self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [], []], status)
        self.assertFalse(os.path.isdir(sub_directory))
        self.assertFalse(os.path.isdir(os.path.dirname(sub_directory)))

    def _commit_something_wrong(self):
        with open(self._foo_path, 'a') as f:
            f.write('something wrong')
        porcelain.add(self.repo, paths=[self._foo_path])
        return porcelain.commit(self.repo, message=b'I may added something wrong', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')

    def test_checkout_to_commit_sha(self):
        self._commit_something_wrong()
        porcelain.checkout_branch(self.repo, self._sha)
        self.assertEqual(self._sha, self.repo.head())

    def test_checkout_to_head(self):
        new_sha = self._commit_something_wrong()
        porcelain.checkout_branch(self.repo, b'HEAD')
        self.assertEqual(new_sha, self.repo.head())

    def _checkout_remote_branch(self):
        errstream = BytesIO()
        outstream = BytesIO()
        porcelain.commit(repo=self.repo.path, message=b'init', author=b'author <email>', committer=b'committer <email>')
        clone_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, clone_path)
        target_repo = porcelain.clone(self.repo.path, target=clone_path, errstream=errstream)
        try:
            self.assertEqual(target_repo[b'HEAD'], self.repo[b'HEAD'])
        finally:
            target_repo.close()
        handle, fullpath = tempfile.mkstemp(dir=clone_path)
        os.close(handle)
        porcelain.add(repo=clone_path, paths=[fullpath])
        porcelain.commit(repo=clone_path, message=b'push', author=b'author <email>', committer=b'committer <email>')
        refs_path = b'refs/heads/foo'
        new_id = self.repo[b'HEAD'].id
        self.assertNotEqual(new_id, ZERO_SHA)
        self.repo.refs[refs_path] = new_id
        porcelain.push(clone_path, 'origin', b'HEAD:' + refs_path, outstream=outstream, errstream=errstream)
        self.assertEqual(target_repo.refs[b'refs/remotes/origin/foo'], target_repo.refs[b'HEAD'])
        porcelain.checkout_branch(target_repo, b'origin/foo')
        original_id = target_repo[b'HEAD'].id
        uni_id = target_repo[b'refs/remotes/origin/uni'].id
        expected_refs = {b'HEAD': original_id, b'refs/heads/master': original_id, b'refs/heads/foo': original_id, b'refs/remotes/origin/foo': original_id, b'refs/remotes/origin/uni': uni_id, b'refs/remotes/origin/HEAD': new_id, b'refs/remotes/origin/master': new_id}
        self.assertEqual(expected_refs, target_repo.get_refs())
        return target_repo

    def test_checkout_remote_branch(self):
        repo = self._checkout_remote_branch()
        repo.close()

    def test_checkout_remote_branch_then_master_then_remote_branch_again(self):
        target_repo = self._checkout_remote_branch()
        self.assertEqual(b'foo', porcelain.active_branch(target_repo))
        _commit_file_with_content(target_repo, 'bar', 'something\n')
        self.assertTrue(os.path.isfile(os.path.join(target_repo.path, 'bar')))
        porcelain.checkout_branch(target_repo, b'master')
        self.assertEqual(b'master', porcelain.active_branch(target_repo))
        self.assertFalse(os.path.isfile(os.path.join(target_repo.path, 'bar')))
        porcelain.checkout_branch(target_repo, b'origin/foo')
        self.assertEqual(b'foo', porcelain.active_branch(target_repo))
        self.assertTrue(os.path.isfile(os.path.join(target_repo.path, 'bar')))
        target_repo.close()