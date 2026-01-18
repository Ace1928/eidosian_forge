import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestAddNonUtf8InBranch(TestCaseForGenericProcessor):

    def file_command_iter(self):

        def command_list():
            committer_a = [b'', b'a@elmer.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(b'foo\x83', kind_to_mode('file', False), None, b'content A\n')
            yield commands.CommitCommand(b'head', b'1', None, committer_a, b'commit 1', None, [], files_one)
        return command_list

    def test_add(self):
        handler, branch = self.get_handler()
        handler.process(self.file_command_iter())
        branch.lock_read()
        self.addCleanup(branch.unlock)
        rev_a = branch.last_revision()
        rtree_a = branch.repository.revision_tree(rev_a)
        self.assertEqual(rev_a, rtree_a.get_file_revision('foo�'))