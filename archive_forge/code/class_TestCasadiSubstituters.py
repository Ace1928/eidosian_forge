import json
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
from pyomo.dae.simulator import (
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.fileutils import import_file
import os
from os.path import abspath, dirname, normpath, join
@unittest.skipIf(not casadi_available, 'Casadi is not available')
class TestCasadiSubstituters(unittest.TestCase):
    """
    Class for testing the Expression substituters for creating valid CasADi
    expressions
    """

    def setUp(self):
        """
        Setting up the testing model
        """
        self.m = m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 10))
        m.v = Var(m.t)
        m.dv = DerivativeVar(m.v)

    def test_substitute_casadi_sym(self):
        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)
        e = m.dv[t] + m.v[t] + m.y + t
        templatemap = {}
        e2 = substitute_pyomo2casadi(e, templatemap)
        self.assertEqual(len(templatemap), 2)
        self.assertIs(type(e2.arg(0)), casadi.SX)
        self.assertIs(type(e2.arg(1)), casadi.SX)
        self.assertIsNot(type(e2.arg(2)), casadi.SX)
        self.assertIs(type(e2.arg(3)), IndexTemplate)
        m.del_component('y')

    def test_substitute_casadi_intrinsic1(self):
        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)
        e = m.v[t]
        templatemap = {}
        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(type(e3), casadi.SX)
        m.del_component('y')

    def test_substitute_casadi_intrinsic2(self):
        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)
        e = sin(m.dv[t]) + log(m.v[t]) + sqrt(m.y) + m.v[t] + t
        templatemap = {}
        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(e3.arg(0)._fcn, casadi.sin)
        self.assertIs(e3.arg(1)._fcn, casadi.log)
        self.assertIs(e3.arg(2)._fcn, casadi.sqrt)
        m.del_component('y')

    def test_substitute_casadi_intrinsic3(self):
        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)
        e = sin(m.dv[t] + m.v[t]) + log(m.v[t] * m.y + m.dv[t] ** 2)
        templatemap = {}
        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(e3.arg(0)._fcn, casadi.sin)
        self.assertIs(e3.arg(1)._fcn, casadi.log)
        m.del_component('y')

    def test_substitute_casadi_intrinsic4(self):
        m = self.m
        m.y = Var()
        t = IndexTemplate(m.t)
        e = m.v[t] * sin(m.dv[t] + m.v[t]) * t
        templatemap = {}
        e3 = substitute_pyomo2casadi(e, templatemap)
        self.assertIs(type(e3.arg(0).arg(0)), casadi.SX)
        self.assertIs(e3.arg(0).arg(1)._fcn, casadi.sin)
        self.assertIs(type(e3.arg(1)), IndexTemplate)
        m.del_component('y')