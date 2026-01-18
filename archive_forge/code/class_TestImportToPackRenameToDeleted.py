import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackRenameToDeleted(TestCaseForGenericProcessor):
    """Test rename to a destination path deleted in this commit."""

    def get_command_iter(self, old_path, new_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(old_path, kind_to_mode(kind, False), None, b'aaa')
                yield commands.FileModifyCommand(new_path, kind_to_mode(kind, False), None, b'bbb')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileDeleteCommand(new_path)
                yield commands.FileRenameCommand(old_path, new_path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_rename_to_deleted_file_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(old_path,), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertContent(branch, revtree1, old_path, b'aaa')
        self.assertContent(branch, revtree1, new_path, b'bbb')
        self.assertContent(branch, revtree2, new_path, b'aaa')
        self.assertRevisionRoot(revtree1, old_path)
        self.assertRevisionRoot(revtree1, new_path)

    def test_rename_to_deleted_symlink_in_root(self):
        handler, branch = self.get_handler()
        old_path = b'a'
        new_path = b'b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(old_path,), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertSymlinkTarget(branch, revtree1, old_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, new_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, new_path, 'aaa')
        self.assertRevisionRoot(revtree1, old_path)
        self.assertRevisionRoot(revtree1, new_path)

    def test_rename_to_deleted_file_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'd/a'
        new_path = b'd/b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd',), (old_path,), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertContent(branch, revtree1, old_path, b'aaa')
        self.assertContent(branch, revtree1, new_path, b'bbb')
        self.assertContent(branch, revtree2, new_path, b'aaa')

    def test_rename_to_deleted_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        old_path = b'd/a'
        new_path = b'd/b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd',), (old_path,), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertSymlinkTarget(branch, revtree1, old_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, new_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, new_path, 'aaa')

    def test_rename_to_deleted_file_in_new_dir(self):
        handler, branch = self.get_handler()
        old_path = b'd1/a'
        new_path = b'd2/b'
        handler.process(self.get_command_iter(old_path, new_path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd1',), (old_path,), (b'd2',), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'd1',), (new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertContent(branch, revtree1, old_path, b'aaa')
        self.assertContent(branch, revtree1, new_path, b'bbb')
        self.assertContent(branch, revtree2, new_path, b'aaa')

    def test_rename_to_deleted_symlink_in_new_dir(self):
        handler, branch = self.get_handler()
        old_path = b'd1/a'
        new_path = b'd2/b'
        handler.process(self.get_command_iter(old_path, new_path, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'd1',), (old_path,), (b'd2',), (new_path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'd1',), (new_path,)], expected_renamed=[(old_path, new_path)])
        self.assertSymlinkTarget(branch, revtree1, old_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree1, new_path, 'bbb')
        self.assertSymlinkTarget(branch, revtree2, new_path, 'aaa')