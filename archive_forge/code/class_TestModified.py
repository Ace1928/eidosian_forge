import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
class TestModified(TestCaseWithTransport):

    def test_modified(self):
        """Test that 'modified' command reports modified files"""
        self._test_modified('a', 'a')

    def test_modified_with_spaces(self):
        """Test that 'modified' command reports modified files with spaces in their names quoted"""
        self._test_modified('a filename with spaces', '"a filename with spaces"')

    def _test_modified(self, name, output):

        def check_modified(expected, null=False):
            command = 'modified'
            if null:
                command += ' --null'
            out, err = self.run_bzr(command)
            self.assertEqual(out, expected)
            self.assertEqual(err, '')
        tree = self.make_branch_and_tree('.')
        check_modified('')
        self.build_tree_contents([(name, b'contents of %s\n' % name.encode('utf-8'))])
        check_modified('')
        tree.add(name)
        check_modified('')
        tree.commit(message='add %s' % output)
        check_modified('')
        self.build_tree_contents([(name, b'changed\n')])
        check_modified(output + '\n')
        check_modified(name + '\x00', null=True)
        tree.commit(message='modified %s' % name)
        check_modified('')

    def test_modified_directory(self):
        """Test --directory option"""
        tree = self.make_branch_and_tree('a')
        self.build_tree(['a/README'])
        tree.add('README')
        tree.commit('r1')
        self.build_tree_contents([('a/README', b'changed\n')])
        out, err = self.run_bzr(['modified', '--directory=a'])
        self.assertEqual('README\n', out)