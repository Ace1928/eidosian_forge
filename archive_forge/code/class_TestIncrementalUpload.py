import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestIncrementalUpload(tests.TestCaseWithTransport, TestUploadMixin):
    do_upload = TestUploadMixin.do_incremental_upload

    def test_delete_one_file(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload()
        self.delete_any('hello')
        self.assertUpFileEqual(b'foo', 'hello')
        self.do_upload()
        self.assertUpPathDoesNotExist('hello')

    def test_delete_dir_and_subdir(self):
        self.make_branch_and_working_tree()
        self.add_dir('dir')
        self.add_dir('dir/subdir')
        self.add_file('dir/subdir/a', b'foo')
        self.do_full_upload()
        self.rename_any('dir/subdir/a', 'a')
        self.delete_any('dir/subdir')
        self.delete_any('dir')
        self.assertUpFileEqual(b'foo', 'dir/subdir/a')
        self.do_upload()
        self.assertUpPathDoesNotExist('dir/subdir/a')
        self.assertUpPathDoesNotExist('dir/subdir')
        self.assertUpPathDoesNotExist('dir')
        self.assertUpFileEqual(b'foo', 'a')

    def test_delete_one_file_rename_to_deleted(self):
        self.make_branch_and_working_tree()
        self.add_file('a', b'foo')
        self.add_file('b', b'bar')
        self.do_full_upload()
        self.delete_any('a')
        self.rename_any('b', 'a')
        self.assertUpFileEqual(b'foo', 'a')
        self.do_upload()
        self.assertUpPathDoesNotExist('b')
        self.assertUpFileEqual(b'bar', 'a')

    def test_rename_outside_dir_delete_dir(self):
        self.make_branch_and_working_tree()
        self.add_dir('dir')
        self.add_file('dir/a', b'foo')
        self.do_full_upload()
        self.rename_any('dir/a', 'a')
        self.delete_any('dir')
        self.assertUpFileEqual(b'foo', 'dir/a')
        self.do_upload()
        self.assertUpPathDoesNotExist('dir/a')
        self.assertUpPathDoesNotExist('dir')
        self.assertUpFileEqual(b'foo', 'a')

    def test_delete_symlink(self):
        self.make_branch_and_working_tree()
        self.add_symlink('link', 'target')
        self.do_full_upload()
        self.delete_any('link')
        self.do_upload()
        self.assertUpPathDoesNotExist('link')

    def test_upload_for_the_first_time_do_a_full_upload(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'bar')
        revid_path = self.tree.branch.get_config_stack().get('upload_revid_location')
        self.assertUpPathDoesNotExist(revid_path)
        self.do_upload()
        self.assertUpFileEqual(b'bar', 'hello')

    def test_ignore_delete_one_file(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'hello')
        self.delete_any('hello')
        self.assertUpFileEqual(b'foo', 'hello')
        self.do_upload()
        self.assertUpFileEqual(b'foo', 'hello')