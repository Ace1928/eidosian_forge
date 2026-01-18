import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackRenameTricky(TestCaseForGenericProcessor):

    def file_command_iter(self, path1, old_path2, new_path2, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path1, kind_to_mode(kind, False), None, b'aaa')
                yield commands.FileModifyCommand(old_path2, kind_to_mode(kind, False), None, b'bbb')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileRenameCommand(old_path2, new_path2)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_rename_file_becomes_directory(self):
        handler, branch = self.get_handler()
        old_path2 = b'foo'
        path1 = b'a/b'
        new_path2 = b'a/b/c'
        handler.process(self.file_command_iter(path1, old_path2, new_path2))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path1,), (old_path2,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path2, new_path2)], expected_kind_changed=[(path1, 'file', 'directory')])
        self.assertContent(branch, revtree1, path1, b'aaa')
        self.assertContent(branch, revtree2, new_path2, b'bbb')

    def test_rename_directory_becomes_file(self):
        handler, branch = self.get_handler()
        old_path2 = b'foo'
        path1 = b'a/b/c'
        new_path2 = b'a/b'
        handler.process(self.file_command_iter(path1, old_path2, new_path2))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (path1,), (old_path2,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path2, new_path2)], expected_removed=[(path1,), (new_path2,)])
        self.assertContent(branch, revtree1, path1, b'aaa')
        self.assertContent(branch, revtree2, new_path2, b'bbb')

    def test_rename_symlink_becomes_directory(self):
        handler, branch = self.get_handler()
        old_path2 = b'foo'
        path1 = b'a/b'
        new_path2 = b'a/b/c'
        handler.process(self.file_command_iter(path1, old_path2, new_path2, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path1,), (old_path2,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path2, new_path2)], expected_kind_changed=[(path1, 'symlink', 'directory')])
        self.assertSymlinkTarget(branch, revtree1, path1, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, new_path2, 'bbb')

    def test_rename_directory_becomes_symlink(self):
        handler, branch = self.get_handler()
        old_path2 = b'foo'
        path1 = b'a/b/c'
        new_path2 = b'a/b'
        handler.process(self.file_command_iter(path1, old_path2, new_path2, 'symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (path1,), (old_path2,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_renamed=[(old_path2, new_path2)], expected_removed=[(path1,), (new_path2,)])
        self.assertSymlinkTarget(branch, revtree1, path1, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, new_path2, 'bbb')