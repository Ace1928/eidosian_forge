import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackModify(TestCaseForGenericProcessor):

    def file_command_iter(self, path, kind='file', content=b'aaa', executable=False, to_kind=None, to_content=b'bbb', to_executable=None):
        if to_kind is None:
            to_kind = kind
        if to_executable is None:
            to_executable = executable
        mode = kind_to_mode(kind, executable)
        to_mode = kind_to_mode(to_kind, to_executable)

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(path, mode, None, content)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileModifyCommand(path, to_mode, None, to_content)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_modify_file_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertContent(branch, revtree2, path, b'bbb')
        self.assertRevisionRoot(revtree1, path)
        self.assertRevisionRoot(revtree2, path)

    def test_modify_file_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertContent(branch, revtree2, path, b'bbb')

    def test_modify_symlink_in_root(self):
        handler, branch = self.get_handler()
        path = b'a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, path, 'bbb')
        self.assertRevisionRoot(revtree1, path)
        self.assertRevisionRoot(revtree2, path)

    def test_modify_symlink_in_subdir(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertSymlinkTarget(branch, revtree2, path, 'bbb')

    def test_modify_file_becomes_symlink(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='file', to_kind='symlink'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_kind_changed=[(path, 'file', 'symlink')])
        self.assertContent(branch, revtree1, path, b'aaa')
        self.assertSymlinkTarget(branch, revtree2, path, 'bbb')

    def test_modify_symlink_becomes_file(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, kind='symlink', to_kind='file'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_kind_changed=[(path, 'symlink', 'file')])
        self.assertSymlinkTarget(branch, revtree1, path, 'aaa')
        self.assertContent(branch, revtree2, path, b'bbb')

    def test_modify_file_now_executable(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, executable=False, to_executable=True, to_content=b'aaa'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertExecutable(branch, revtree1, path, False)
        self.assertExecutable(branch, revtree2, path, True)

    def test_modify_file_no_longer_executable(self):
        handler, branch = self.get_handler()
        path = b'a/a'
        handler.process(self.file_command_iter(path, executable=True, to_executable=False, to_content=b'aaa'))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (path,)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_modified=[(path,)])
        self.assertExecutable(branch, revtree1, path, True)
        self.assertExecutable(branch, revtree2, path, False)