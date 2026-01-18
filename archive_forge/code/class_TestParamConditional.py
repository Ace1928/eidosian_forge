import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
class TestParamConditional(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def tearDown(self):
        self.model = None

    def test_if_const_param_1value(self):
        self.model.p = Param(initialize=1.0)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo numeric value \\(p\\) to bool'):
            if self.model.p:
                pass
        instance = self.model.create_instance()
        if instance.p:
            pass
        else:
            self.fail('Wrong condition value')

    def test_if_const_param_0value(self):
        self.model.p = Param(initialize=0.0)
        with self.assertRaisesRegex(PyomoException, 'Cannot convert non-constant Pyomo numeric value \\(p\\) to bool'):
            if self.model.p:
                pass
        instance = self.model.create_instance()
        if instance.p:
            self.fail('Wrong condition value')
        else:
            pass