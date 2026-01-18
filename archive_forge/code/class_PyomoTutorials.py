import runpy
import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import pyutilib, pyutilib_available
class PyomoTutorials(unittest.TestCase):

    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp_path = list(sys.path)
        os.chdir(tutorial_dir)
        sys.path = [os.path.dirname(tutorial_dir)] + sys.path
        sys.path.append(os.path.dirname(tutorial_dir))
        sys.stderr.flush()
        sys.stdout.flush()
        self.save_stdout = sys.stdout
        self.save_stderr = sys.stderr

    def tearDown(self):
        os.chdir(self.cwd)
        sys.path = self.tmp_path
        sys.stdout = self.save_stdout
        sys.stderr = self.save_stderr

    def driver(self, name):
        OUTPUT = open(currdir + name + '.log', 'w')
        sys.stdout = OUTPUT
        sys.stderr = OUTPUT
        runpy.run_module(name, None, '__main__')
        OUTPUT.close()
        self.assertIn(open(tutorial_dir + name + '.out', 'r').read(), open(currdir + name + '.log', 'r').read())
        os.remove(currdir + name + '.log')

    def test_data(self):
        self.driver('data')

    @unittest.skipIf(not (_xlrd or _openpyxl), 'Cannot read excel file.')
    @unittest.skipIf(not (_win32com and _excel_available and pyutilib_available), 'Cannot read excel file.')
    def test_excel(self):
        self.driver('excel')

    def test_set(self):
        self.driver('set')

    def test_table(self):
        self.driver('table')

    def test_param(self):
        self.driver('param')