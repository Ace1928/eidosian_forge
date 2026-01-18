import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveTextConflicts(TestParametrizedResolveConflicts):
    _conflict_type = bzr_conflicts.TextConflict
    _path = None
    _file_id = None
    scenarios = mirror_scenarios([(dict(_base_actions='create_file', _path='file', _file_id=b'file-id'), ('filed_modified_A', dict(actions='modify_file_A', check='file_has_content_A')), ('file_modified_B', dict(actions='modify_file_B', check='file_has_content_B'))), (dict(_base_actions='create_file_in_dir', _path='dir/file', _file_id=b'file-id'), ('filed_modified_A_in_dir', dict(actions='modify_file_A_in_dir', check='file_in_dir_has_content_A')), ('file_modified_B', dict(actions='modify_file_B_in_dir', check='file_in_dir_has_content_B')))])

    def do_create_file(self, path='file'):
        return [('add', (path, b'file-id', 'file', b'trunk content\n'))]

    def do_modify_file_A(self):
        return [('modify', ('file', b'trunk content\nfeature A\n'))]

    def do_modify_file_B(self):
        return [('modify', ('file', b'trunk content\nfeature B\n'))]

    def do_modify_file_A_in_dir(self):
        return [('modify', ('dir/file', b'trunk content\nfeature A\n'))]

    def do_modify_file_B_in_dir(self):
        return [('modify', ('dir/file', b'trunk content\nfeature B\n'))]

    def check_file_has_content_A(self, path='file'):
        self.assertFileEqual(b'trunk content\nfeature A\n', osutils.pathjoin('branch', path))

    def check_file_has_content_B(self, path='file'):
        self.assertFileEqual(b'trunk content\nfeature B\n', osutils.pathjoin('branch', path))

    def do_create_file_in_dir(self):
        return [('add', ('dir', b'dir-id', 'directory', ''))] + self.do_create_file('dir/file')

    def check_file_in_dir_has_content_A(self):
        self.check_file_has_content_A('dir/file')

    def check_file_in_dir_has_content_B(self):
        self.check_file_has_content_B('dir/file')

    def _get_resolve_path_arg(self, wt, action):
        return self._path

    def assertTextConflict(self, wt, c):
        self.assertEqual(self._file_id, c.file_id)
        self.assertEqual(self._path, c.path)
    _assert_conflict = assertTextConflict