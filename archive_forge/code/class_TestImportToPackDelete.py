import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDelete(TestCaseForGenericProcessor):

    def file_command_iter(self, path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path, kind_to_mode(kind, False), None, b'aaa')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileDeleteCommand(path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_delete_file_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertRevisionRoot(revtree1, path)

    def test_delete_file_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a',), (path,)])
        self.assertContent(branch, revtree1, path, b'aaa')

    def test_delete_symlink_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertRevisionRoot(revtree1, path)

    def test_delete_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a',), (path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')

    def test_delete_file_in_deep_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/b/c/d'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a',), (b'a/b',), (b'a/b/c',), (path,)])
        self.assertContent(branch, revtree1, path, b'aaa')