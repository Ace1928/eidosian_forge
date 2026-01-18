import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDeleteNew(TestCaseForGenericProcessor):
    """Test deletion of a newly added file."""

    def file_command_iter(self, path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path, kind_to_mode(kind, False), None, b'aaa')
                yield commands.FileDeleteCommand(path)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)
        return command_list

    def test_delete_new_file_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1)

    def test_delete_new_file_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1)

    def test_delete_new_symlink_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1)

    def test_delete_new_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1)

    def test_delete_new_file_in_deep_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/b/c/d'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1)