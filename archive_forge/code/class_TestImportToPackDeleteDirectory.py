import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportToPackDeleteDirectory(TestCaseForGenericProcessor):

    def file_command_iter(self, paths, dir):

        def command_list():
            author = [b'', b'bugs@a.com', time.time(), time.timezone]
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]

            def files_one():
                for i, path in enumerate(paths):
                    yield commands.FileModifyCommand(path, kind_to_mode('file', False), None, b'aaa%d' % i)
            yield commands.CommitCommand(b'head', b'1', author, committer, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileDeleteCommand(dir)
            yield commands.CommitCommand(b'head', b'2', author, committer, b'commit 2', b':1', [], files_two)
        return command_list

    def test_delete_dir(self):
        handler, branch = self.get_handler()
        paths = [b'a/b/c', b'a/b/d', b'a/b/e/f', b'a/g']
        dir = b'a/b'
        handler.process(self.file_command_iter(paths, dir))
        revtree0, revtree1 = self.assertChanges(branch, 1, expected_added=[(b'a',), (b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/e',), (b'a/b/e/f',), (b'a/g',)])
        revtree1, revtree2 = self.assertChanges(branch, 2, expected_removed=[(b'a/b',), (b'a/b/c',), (b'a/b/d',), (b'a/b/e',), (b'a/b/e/f',)])