import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
class TestUploadMixin(UploadUtilsMixin):
    """Helper class to share tests between full and incremental uploads."""

    def _test_create_file(self, file_name):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file(file_name, b'foo')
        self.do_upload()
        self.assertUpFileEqual(b'foo', file_name)

    def test_create_file(self):
        self._test_create_file('hello')

    def test_unicode_create_file(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_create_file('hellØ')

    def _test_create_file_in_dir(self, dir_name, file_name):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_dir(dir_name)
        fpath = '{}/{}'.format(dir_name, file_name)
        self.add_file(fpath, b'baz')
        self.assertUpPathDoesNotExist(fpath)
        self.do_upload()
        self.assertUpFileEqual(b'baz', fpath)
        self.assertUpPathModeEqual(dir_name, 509)

    def test_create_file_in_dir(self):
        self._test_create_file_in_dir('dir', 'goodbye')

    def test_unicode_create_file_in_dir(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_create_file_in_dir('dirØ', 'goodbyeØ')

    def test_modify_file(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload()
        self.modify_file('hello', b'bar')
        self.assertUpFileEqual(b'foo', 'hello')
        self.do_upload()
        self.assertUpFileEqual(b'bar', 'hello')

    def _test_rename_one_file(self, old_name, new_name):
        self.make_branch_and_working_tree()
        self.add_file(old_name, b'foo')
        self.do_full_upload()
        self.rename_any(old_name, new_name)
        self.assertUpFileEqual(b'foo', old_name)
        self.do_upload()
        self.assertUpFileEqual(b'foo', new_name)

    def test_rename_one_file(self):
        self._test_rename_one_file('hello', 'goodbye')

    def test_unicode_rename_one_file(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_rename_one_file('helloØ', 'goodbyeØ')

    def test_rename_and_change_file(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload()
        self.rename_any('hello', 'goodbye')
        self.modify_file('goodbye', b'bar')
        self.assertUpFileEqual(b'foo', 'hello')
        self.do_upload()
        self.assertUpFileEqual(b'bar', 'goodbye')

    def test_rename_two_files(self):
        self.make_branch_and_working_tree()
        self.add_file('a', b'foo')
        self.add_file('b', b'qux')
        self.do_full_upload()
        self.rename_any('b', 'c')
        self.rename_any('a', 'b')
        self.assertUpFileEqual(b'foo', 'a')
        self.assertUpFileEqual(b'qux', 'b')
        self.do_upload()
        self.assertUpFileEqual(b'foo', 'b')
        self.assertUpFileEqual(b'qux', 'c')

    def test_upload_revision(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file('hello', b'foo')
        self.modify_file('hello', b'bar')
        self.assertUpPathDoesNotExist('hello')
        revspec = revisionspec.RevisionSpec.from_string('2')
        self.do_upload(revision=[revspec])
        self.assertUpFileEqual(b'foo', 'hello')

    def test_no_upload_when_changes(self):
        self.make_branch_and_working_tree()
        self.add_file('a', b'foo')
        self.set_file_content('a', b'bar')
        self.assertRaises(errors.UncommittedChanges, self.do_upload)

    def test_no_upload_when_conflicts(self):
        self.make_branch_and_working_tree()
        self.add_file('a', b'foo')
        self.run_bzr('branch branch other')
        self.modify_file('a', b'bar')
        other_tree = workingtree.WorkingTree.open('other')
        self.set_file_content('a', b'baz', 'other/')
        other_tree.commit('modify file a')
        self.run_bzr('merge -d branch other', retcode=1)
        self.assertRaises(errors.UncommittedChanges, self.do_upload)

    def _test_change_file_into_dir(self, file_name):
        self.make_branch_and_working_tree()
        self.add_file(file_name, b'foo')
        self.do_full_upload()
        self.transform_file_into_dir(file_name)
        fpath = '{}/{}'.format(file_name, 'file')
        self.add_file(fpath, b'bar')
        self.assertUpFileEqual(b'foo', file_name)
        self.do_upload()
        self.assertUpFileEqual(b'bar', fpath)

    def test_change_file_into_dir(self):
        self._test_change_file_into_dir('hello')

    def test_unicode_change_file_into_dir(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_change_file_into_dir('helloØ')

    def test_change_dir_into_file(self):
        self.make_branch_and_working_tree()
        self.add_dir('hello')
        self.add_file('hello/file', b'foo')
        self.do_full_upload()
        self.delete_any('hello/file')
        self.transform_dir_into_file('hello', b'bar')
        self.assertUpFileEqual(b'foo', 'hello/file')
        self.do_upload()
        self.assertUpFileEqual(b'bar', 'hello')

    def _test_make_file_executable(self, file_name):
        self.make_branch_and_working_tree()
        self.add_file(file_name, b'foo')
        self.chmod_file(file_name, 436)
        self.do_full_upload()
        self.chmod_file(file_name, 493)
        self.assertUpPathModeEqual(file_name, 436)
        self.do_upload()
        self.assertUpPathModeEqual(file_name, 509)

    def test_make_file_executable(self):
        self._test_make_file_executable('hello')

    def test_unicode_make_file_executable(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_make_file_executable('helloØ')

    def test_create_symlink(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_symlink('link', 'target')
        self.do_upload()
        self.assertUpPathExists('link')

    def test_rename_symlink(self):
        self.make_branch_and_working_tree()
        old_name, new_name = ('old-link', 'new-link')
        self.add_symlink(old_name, 'target')
        self.do_full_upload()
        self.rename_any(old_name, new_name)
        self.do_upload()
        self.assertUpPathExists(new_name)

    def get_upload_auto(self):
        b = controldir.ControlDir.open(self.tree.basedir).open_branch()
        return b.get_config_stack().get('upload_auto')

    def test_upload_auto(self):
        """Test that upload --auto sets the upload_auto option"""
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.assertFalse(self.get_upload_auto())
        self.do_full_upload(auto=True)
        self.assertUpFileEqual(b'foo', 'hello')
        self.assertTrue(self.get_upload_auto())
        self.add_file('bye', b'bar')
        self.do_full_upload()
        self.assertUpFileEqual(b'bar', 'bye')
        self.assertTrue(self.get_upload_auto())

    def test_upload_noauto(self):
        """Test that upload --no-auto unsets the upload_auto option"""
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload(auto=True)
        self.assertUpFileEqual(b'foo', 'hello')
        self.assertTrue(self.get_upload_auto())
        self.add_file('bye', b'bar')
        self.do_full_upload(auto=False)
        self.assertUpFileEqual(b'bar', 'bye')
        self.assertFalse(self.get_upload_auto())
        self.add_file('again', b'baz')
        self.do_full_upload()
        self.assertUpFileEqual(b'baz', 'again')
        self.assertFalse(self.get_upload_auto())

    def test_upload_from_subdir(self):
        self.make_branch_and_working_tree()
        self.build_tree(['branch/foo/', 'branch/foo/bar'])
        self.tree.add(['foo/', 'foo/bar'])
        self.tree.commit('Add directory')
        self.do_full_upload(directory='branch/foo')

    def test_upload_revid_path_in_dir(self):
        self.make_branch_and_working_tree()
        self.add_dir('dir')
        self.add_file('dir/goodbye', b'baz')
        revid_path = 'dir/revid-path'
        self.tree.branch.get_config_stack().set('upload_revid_location', revid_path)
        self.assertUpPathDoesNotExist(revid_path)
        self.do_full_upload()
        self.add_file('dir/hello', b'foo')
        self.do_upload()
        self.assertUpPathExists(revid_path)
        self.assertUpFileEqual(b'baz', 'dir/goodbye')
        self.assertUpFileEqual(b'foo', 'dir/hello')

    def test_ignore_file(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'foo')
        self.add_file('foo', b'bar')
        self.do_upload()
        self.assertUpPathDoesNotExist('foo')

    def test_ignore_regexp(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'f*')
        self.add_file('foo', b'bar')
        self.do_upload()
        self.assertUpPathDoesNotExist('foo')

    def test_ignore_directory(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'dir')
        self.add_dir('dir')
        self.do_upload()
        self.assertUpPathDoesNotExist('dir')

    def test_ignore_nested_directory(self):
        self.make_branch_and_working_tree()
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'dir')
        self.add_dir('dir')
        self.add_dir('dir/foo')
        self.add_file('dir/foo/bar', b'bar contents')
        self.do_upload()
        self.assertUpPathDoesNotExist('dir')
        self.assertUpPathDoesNotExist('dir/foo/bar')

    def test_ignore_change_file_into_dir(self):
        self.make_branch_and_working_tree()
        self.add_file('hello', b'foo')
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'hello')
        self.transform_file_into_dir('hello')
        self.add_file('hello/file', b'bar')
        self.assertUpFileEqual(b'foo', 'hello')
        self.do_upload()
        self.assertUpFileEqual(b'foo', 'hello')

    def test_ignore_change_dir_into_file(self):
        self.make_branch_and_working_tree()
        self.add_dir('hello')
        self.add_file('hello/file', b'foo')
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'hello')
        self.delete_any('hello/file')
        self.transform_dir_into_file('hello', b'bar')
        self.assertUpFileEqual(b'foo', 'hello/file')
        self.do_upload()
        self.assertUpFileEqual(b'foo', 'hello/file')

    def test_ignore_delete_dir_in_subdir(self):
        self.make_branch_and_working_tree()
        self.add_dir('dir')
        self.add_dir('dir/subdir')
        self.add_file('dir/subdir/a', b'foo')
        self.do_full_upload()
        self.add_file('.bzrignore-upload', b'dir/subdir')
        self.rename_any('dir/subdir/a', 'dir/a')
        self.delete_any('dir/subdir')
        self.assertUpFileEqual(b'foo', 'dir/subdir/a')
        self.do_upload()
        self.assertUpFileEqual(b'foo', 'dir/a')