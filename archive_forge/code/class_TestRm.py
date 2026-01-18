from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestRm(script.TestCaseWithTransportAndScript):

    def test_rm_usage(self):
        self.assertRaises(SyntaxError, self.run_script, '$ rm')
        self.assertRaises(SyntaxError, self.run_script, '$ rm -ff foo')

    def test_rm_file(self):
        self.run_script('$ echo content >file')
        self.assertPathExists('file')
        self.run_script('$ rm file')
        self.assertPathDoesNotExist('file')

    def test_rm_file_force(self):
        self.assertPathDoesNotExist('file')
        self.run_script('$ rm -f file')
        self.assertPathDoesNotExist('file')

    def test_rm_files(self):
        self.run_script('\n$ echo content >file\n$ echo content >file2\n')
        self.assertPathExists('file2')
        self.run_script('$ rm file file2')
        self.assertPathDoesNotExist('file2')

    def test_rm_dir(self):
        self.run_script('$ mkdir dir')
        self.assertPathExists('dir')
        self.run_script("\n$ rm dir\n2>rm: cannot remove 'dir': Is a directory\n")
        self.assertPathExists('dir')

    def test_rm_dir_recursive(self):
        self.run_script('\n$ mkdir dir\n$ rm -r dir\n')
        self.assertPathDoesNotExist('dir')