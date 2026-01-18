import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDeleteMultiLevel(TestCaseForGenericProcessor):

    def file_command_iter(self, paths, paths_to_delete):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                for i, path in enumerate(paths):
                    yield commands.FileModifyCommand(path, kind_to_mode('file', False), None, b'aaa%d' % i)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                for path in paths_to_delete:
                    yield commands.FileDeleteCommand(path)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_delete_files_in_multiple_levels(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d/e']
        paths_to_delete = [b'a/b/c', b'a/b/d/e']
        handler.process(self.file_command_iter(paths, paths_to_delete))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',)])

    def test_delete_file_single_level(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d/e']
        paths_to_delete = [b'a/b/d/e']
        handler.process(self.file_command_iter(paths, paths_to_delete))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b/d',), (b'a/b/d/e',)])

    def test_delete_file_complex_level(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d/e', b'a/f/g', b'a/h', b'a/b/d/i/j']
        paths_to_delete = [b'a/b/c', b'a/b/d/e', b'a/f/g', b'a/b/d/i/j']
        handler.process(self.file_command_iter(paths, paths_to_delete))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',), (b'a/f',), (b'a/f/g',), (b'a/h',), (b'a/b/d/i',), (b'a/b/d/i/j',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/d/e',), (b'a/f',), (b'a/f/g',), (b'a/b/d/i',), (b'a/b/d/i/j',)])