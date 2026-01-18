import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
class InTempDir(object):

    def __init__(self, suffix=None, prefix=None, dir=None):
        self._suffix = suffix
        self._prefix = prefix
        self._dir = dir

    def __enter__(self):
        self._cwd = os.getcwd()
        TempfileManager.push()
        self._tempdir = TempfileManager.create_tempdir(suffix=self._suffix, prefix=self._prefix, dir=self._dir)
        os.chdir(self._tempdir)

    def __exit__(self, ex_type, ex_val, ex_bt):
        os.chdir(self._cwd)
        TempfileManager.pop()