import json
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
import pyomo.common
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import ConcreteModel
from pyomo.opt import ResultsFormat, SolverResults, SolverFactory
def Xtest_options(self):
    """Test ASL options behavior"""
    results = self.asl.solve(currdir + 'bell3a.mps', logfile=currdir + 'test_options.log', options="sec=0.1 foo=1 bar='a=b c=d' xx_zz=yy", suffixes=['.*'])
    results.write(filename=currdir + 'test_options.txt', times=False)
    _out, _log = (join(currdir, 'test_options.txt'), join(currdir, 'test4_asl.txt'))
    self.assertTrue(cmp(_out, _log), msg='Files %s and %s differ' % (_out, _log))