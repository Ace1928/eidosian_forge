import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
class RunAllNEOSSolvers(object):

    def test_bonmin(self):
        self._run('bonmin')

    def test_cbc(self):
        self._run('cbc')

    def test_conopt(self):
        self._run('conopt')

    def test_couenne(self):
        self._run('couenne')

    def test_cplex(self):
        self._run('cplex')

    def test_filmint(self):
        self._run('filmint')

    def test_filter(self):
        self._run('filter')

    def test_ipopt(self):
        self._run('ipopt')

    def test_knitro(self):
        self._run('knitro')

    def test_lbfgsb(self):
        self._run('l-bfgs-b', False)

    def test_lancelot(self):
        self._run('lancelot')

    def test_loqo(self):
        self._run('loqo')

    def test_minlp(self):
        self._run('minlp')

    def test_minos(self):
        self._run('minos')

    def test_minto(self):
        self._run('minto')

    def test_mosek(self):
        self._run('mosek')

    def test_octeract(self):
        self._run('octeract')

    def test_ooqp(self):
        if self.sense == pyo.maximize:
            with self.assertRaisesRegex(AssertionError, '.* != 1 within'):
                self._run('ooqp')
        else:
            self._run('ooqp')

    def test_snopt(self):
        self._run('snopt')

    def test_raposa(self):
        self._run('raposa')

    def test_lgo(self):
        self._run('lgo')