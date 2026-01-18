import os
import sys
import tempfile
from .. import mergetools, tests
class TestFilenameSubstitution(tests.TestCaseInTempDir):

    def test_simple_filename(self):
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        args, tmpfile = mergetools._subst_filename(cmd_list, 'test.txt')
        self.assertEqual(['kdiff3', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER', '-o', 'test.txt'], args)

    def test_spaces(self):
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        args, tmpfile = mergetools._subst_filename(cmd_list, 'file with space.txt')
        self.assertEqual(['kdiff3', 'file with space.txt.BASE', 'file with space.txt.THIS', 'file with space.txt.OTHER', '-o', 'file with space.txt'], args)

    def test_spaces_and_quotes(self):
        cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
        args, tmpfile = mergetools._subst_filename(cmd_list, 'file with "space and quotes".txt')
        self.assertEqual(['kdiff3', 'file with "space and quotes".txt.BASE', 'file with "space and quotes".txt.THIS', 'file with "space and quotes".txt.OTHER', '-o', 'file with "space and quotes".txt'], args)

    def test_tempfile(self):
        self.build_tree(('test.txt', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER'))
        cmd_list = ['some_tool', '{this_temp}']
        args, tmpfile = mergetools._subst_filename(cmd_list, 'test.txt')
        self.assertPathExists(tmpfile)
        os.remove(tmpfile)