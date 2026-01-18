import os
import sys
from os.path import abspath, dirname, normpath, join
from pyomo.common.fileutils import import_file
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import pyomo.opt
from pyomo.environ import SolverFactory, TransformationFactory
class Reformulate(unittest.TestCase, CommonTests):
    solve = False

    def tearDown(self):
        if os.path.exists(os.path.join(currdir, 'result.yml')):
            os.remove(os.path.join(currdir, 'result.yml'))

    def pyomo(self, *args, **kwds):
        args = list(args)
        args.append('--output=' + self.problem + '_result.lp')
        CommonTests.pyomo(self, *args, **kwds)

    def referenceFile(self, problem, solver):
        return join(currdir, problem + '_' + solver + '.lp')

    def check(self, problem, solver):
        self.assertEqual(*load_and_compare_lp_baseline(self.referenceFile(problem, solver), join(currdir, self.problem + '_result.lp')))
        if os.path.exists(join(currdir, self.problem + '_result.lp')):
            os.remove(join(currdir, self.problem + '_result.lp'))