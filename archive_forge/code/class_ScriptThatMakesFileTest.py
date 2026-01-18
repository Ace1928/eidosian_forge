import reportlab
import sys, os, fnmatch, re, functools
from configparser import ConfigParser
import unittest
from reportlab.lib.utils import isCompactDistro, __rl_loader__, rl_isdir, asUnicode
class ScriptThatMakesFileTest(unittest.TestCase):
    """Runs a Python script at OS level, expecting it to produce a file.

    It CDs to the working directory to run the script."""

    def __init__(self, scriptDir, scriptName, outFileName, verbose=0):
        self.scriptDir = scriptDir
        self.scriptName = scriptName
        self.outFileName = outFileName
        self.verbose = verbose
        unittest.TestCase.__init__(self)

    def setUp(self):
        self.cwd = os.getcwd()
        global testsFolder
        scriptDir = self.scriptDir
        if not os.path.isabs(scriptDir):
            scriptDir = os.path.join(testsFolder, scriptDir)
        os.chdir(scriptDir)
        assert os.path.isfile(self.scriptName), 'Script %s not found!' % self.scriptName
        if os.path.isfile(self.outFileName):
            os.remove(self.outFileName)

    def tearDown(self):
        os.chdir(self.cwd)

    def runTest(self):
        fmt = sys.platform == 'win32' and '"%s" %s' or '%s %s'
        import subprocess
        out = subprocess.check_output((sys.executable, self.scriptName))
        if self.verbose:
            print(out)
        assert os.path.isfile(self.outFileName), 'File %s not created!' % self.outFileName