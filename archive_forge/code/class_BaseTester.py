from itertools import zip_longest
import json
import re
import os
import sys
from os.path import abspath, dirname, join
from filecmp import cmp
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
from io import StringIO
class BaseTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk')

    def pyomo(self, cmd, **kwds):
        if 'root' in kwds:
            OUTPUT = kwds['root'] + '.out'
            results = kwds['root'] + '.jsn'
            TempfileManager.add_tempfile(OUTPUT, exists=False)
            TempfileManager.add_tempfile(results, exists=False)
        else:
            OUTPUT = StringIO()
            results = 'results.jsn'
            TempfileManager.create_tempfile(suffix='results.jsn')
        with capture_output(OUTPUT):
            try:
                _dir = os.getcwd()
                os.chdir(currdir)
                args = ['solve', '--solver=glpk', '--results-format=json', '--save-results=%s' % results]
                if type(cmd) is list:
                    args.extend(cmd)
                elif cmd.endswith('json') or cmd.endswith('yaml'):
                    args.append(cmd)
                else:
                    args.extend(re.split('[ ]+', cmd))
                output = main.main(args)
            finally:
                os.chdir(_dir)
        if not 'root' in kwds:
            return OUTPUT.getvalue()
        return output

    def setUp(self):
        if not 'glpk' in solvers:
            self.skipTest('GLPK is not installed')
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def run_pyomo(self, cmd, root):
        results = root + '.jsn'
        TempfileManager.add_tempfile(results, exists=False)
        output = root + '.out'
        TempfileManager.add_tempfile(output, exists=False)
        cmd = ['pyomo', 'solve', '--solver=glpk', '--results-format=json', '--save-results=%s' % results] + cmd
        with open(output, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=f)
        return result