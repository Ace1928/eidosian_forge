import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackCopy(TestCaseForGenericProcessor):

    def file_command_iter(self, src_path, dest_path, kind='file'):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(src_path, kind_to_mode(kind, False), None, b'aaa')
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileCopyCommand(src_path, dest_path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_copy_file_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'aaa')
        self.assertContent(branch, revtree2, dest_path, b'aaa')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree2, dest_path)

    def test_copy_file_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'a/b'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'aaa')
        self.assertContent(branch, revtree2, dest_path, b'aaa')

    def test_copy_file_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'b/a'
        handler.process(self.file_command_iter(src_path, dest_path))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(b'b',), (dest_path,)])
        self.assertContent(branch, revtree1, src_path, b'aaa')
        self.assertContent(branch, revtree2, src_path, b'aaa')
        self.assertContent(branch, revtree2, dest_path, b'aaa')

    def test_copy_symlink_in_root(self):
        handler, branch = self.get_handler()
        src_path = b'a'
        dest_path = b'b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'aaa')
        self.assertRevisionRoot(revtree1, src_path)
        self.assertRevisionRoot(revtree2, dest_path)

    def test_copy_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'a/b'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'aaa')

    def test_copy_symlink_to_new_dir(self):
        handler, branch = self.get_handler()
        src_path = b'a/a'
        dest_path = b'b/a'
        handler.process(self.file_command_iter(src_path, dest_path, 'symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_added=[(b'b',), (dest_path,)])
        self.assertSymlinkTarget(branch, revtree1, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, src_path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, dest_path, 'aaa')