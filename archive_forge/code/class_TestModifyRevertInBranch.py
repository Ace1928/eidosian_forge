import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestModifyRevertInBranch(TestCaseForGenericProcessor):

    def file_command_iter(self):

        def command_list():
            committer_a = [b'', b'a@elmer.com', time.time(), time.timezone]
            committer_b = [b'', b'b@elmer.com', time.time(), time.timezone]
            committer_c = [b'', b'c@elmer.com', time.time(), time.timezone]
            committer_d = [b'', b'd@elmer.com', time.time(), time.timezone]

            def files_one():
                yield commands.FileModifyCommand(b'foo', kind_to_mode('file', False), None, b'content A\n')
            yield commands.CommitCommand(b'head', b'1', None, committer_a, b'commit 1', None, [], files_one)

            def files_two():
                yield commands.FileModifyCommand(b'foo', kind_to_mode('file', False), None, b'content B\n')
            yield commands.CommitCommand(b'head', b'2', None, committer_b, b'commit 2', b':1', [], files_two)

            def files_three():
                yield commands.FileModifyCommand(b'foo', kind_to_mode('file', False), None, b'content A\n')
            yield commands.CommitCommand(b'head', b'3', None, committer_c, b'commit 3', b':2', [], files_three)
            yield commands.CommitCommand(b'head', b'4', None, committer_d, b'commit 4', b':1', [b':3'], lambda: [])
        return command_list

    def test_modify_revert(self):
        handler, branch = self.get_handler()
        handler.process(self.file_command_iter())
        branch.lock_read()
        self.addCleanup(branch.unlock)
        rev_d = branch.last_revision()
        rev_a, rev_c = branch.repository.get_parent_map([rev_d])[rev_d]
        rev_b = branch.repository.get_parent_map([rev_c])[rev_c][0]
        rtree_a, rtree_b, rtree_c, rtree_d = branch.repository.revision_trees([rev_a, rev_b, rev_c, rev_d])
        self.assertEqual(rev_a, rtree_a.get_file_revision('foo'))
        self.assertEqual(rev_b, rtree_b.get_file_revision('foo'))
        self.assertEqual(rev_c, rtree_c.get_file_revision('foo'))
        self.assertEqual(rev_c, rtree_d.get_file_revision('foo'))