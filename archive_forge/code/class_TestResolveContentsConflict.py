import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveContentsConflict(TestParametrizedResolveConflicts):
    _conflict_type = bzr_conflicts.ContentsConflict
    _path = None
    _file_id = None
    scenarios = mirror_scenarios([(dict(_base_actions='create_file', _path='file', _file_id=b'file-id'), ('file_modified', dict(actions='modify_file', check='file_has_more_content')), ('file_deleted', dict(actions='delete_file', check='file_doesnt_exist'))), (dict(_base_actions='create_file', _path='new-file', _file_id=b'file-id'), ('file_renamed_and_modified', dict(actions='modify_and_rename_file', check='file_renamed_and_more_content')), ('file_deleted', dict(actions='delete_file', check='file_doesnt_exist'))), (dict(_base_actions='create_file_in_dir', _path='dir/file', _file_id=b'file-id'), ('file_modified_in_dir', dict(actions='modify_file_in_dir', check='file_in_dir_has_more_content')), ('file_deleted_in_dir', dict(actions='delete_file_in_dir', check='file_in_dir_doesnt_exist')))])

    def do_create_file(self):
        return [('add', ('file', b'file-id', 'file', b'trunk content\n'))]

    def do_modify_file(self):
        return [('modify', ('file', b'trunk content\nmore content\n'))]

    def do_modify_and_rename_file(self):
        return [('modify', ('new-file', b'trunk content\nmore content\n')), ('rename', ('file', 'new-file'))]

    def check_file_has_more_content(self):
        self.assertFileEqual(b'trunk content\nmore content\n', 'branch/file')

    def check_file_renamed_and_more_content(self):
        self.assertFileEqual(b'trunk content\nmore content\n', 'branch/new-file')

    def do_delete_file(self):
        return [('unversion', 'file')]

    def do_delete_file_in_dir(self):
        return [('unversion', 'dir/file')]

    def check_file_doesnt_exist(self):
        self.assertPathDoesNotExist('branch/file')

    def do_create_file_in_dir(self):
        return [('add', ('dir', b'dir-id', 'directory', '')), ('add', ('dir/file', b'file-id', 'file', b'trunk content\n'))]

    def do_modify_file_in_dir(self):
        return [('modify', ('dir/file', b'trunk content\nmore content\n'))]

    def check_file_in_dir_has_more_content(self):
        self.assertFileEqual(b'trunk content\nmore content\n', 'branch/dir/file')

    def check_file_in_dir_doesnt_exist(self):
        self.assertPathDoesNotExist('branch/dir/file')

    def _get_resolve_path_arg(self, wt, action):
        return self._path

    def assertContentsConflict(self, wt, c):
        self.assertEqual(self._file_id, c.file_id)
        self.assertEqual(self._path, c.path)
    _assert_conflict = assertContentsConflict