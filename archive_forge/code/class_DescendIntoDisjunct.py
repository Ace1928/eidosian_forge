import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
class DescendIntoDisjunct(unittest.TestCase):

    def get_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-100, 100))
        m.obj = Objective(expr=m.x)
        m.disjunct1 = Disjunct()
        m.disjunct1.comp = Complementarity(expr=complements(m.x >= 0, 4 * m.x - 3 >= 0))
        m.disjunct2 = Disjunct()
        m.disjunct2.cons = Constraint(expr=m.x >= 2)
        m.disjunction = Disjunction(expr=[m.disjunct1, m.disjunct2])
        return m

    def check_simple_disjunction(self, m):
        compBlock = m.disjunct1.component('comp')
        self.assertIsInstance(compBlock, Block)
        self.assertIsInstance(compBlock.component('expr1'), Disjunct)
        self.assertIsInstance(compBlock.component('expr2'), Disjunct)
        self.assertIsInstance(compBlock.component('complements'), Disjunction)

    def test_simple_disjunction_descend_into_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.simple_disjunction').apply_to(m)
        self.check_simple_disjunction(m)

    def test_simple_disjunction_on_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.simple_disjunction').apply_to(m.disjunct1)
        self.check_simple_disjunction(m)

    def check_simple_nonlinear(self, m):
        compBlock = m.disjunct1.component('comp')
        self.assertIsInstance(compBlock, Block)
        self.assertIsInstance(compBlock.component('v'), Var)
        self.assertIsInstance(compBlock.component('c'), Constraint)
        self.assertIsInstance(compBlock.component('ccon'), Constraint)
        self.assertIsInstance(compBlock.component('ve'), Constraint)

    def test_simple_nonlinear_descend_into_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.simple_nonlinear').apply_to(m)
        self.check_simple_nonlinear(m)

    def test_simple_nonlinear_on_disjunct(self):
        m = self.get_model()

    def check_standard_form(self, m):
        compBlock = m.disjunct1.component('comp')
        self.assertIsInstance(compBlock, Block)
        self.assertIsInstance(compBlock.component('v'), Var)
        self.assertIsInstance(compBlock.component('c'), Constraint)
        self.assertIsInstance(compBlock.component('ve'), Constraint)

    def test_standard_form_descend_into_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.standard_form').apply_to(m)
        self.check_standard_form(m)

    def test_standard_form_on_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.standard_form').apply_to(m.disjunct1)
        self.check_standard_form(m)

    def check_nl(self, m):
        compBlock = m.disjunct1.component('comp')
        self.assertIsInstance(compBlock, Block)
        self.assertIsInstance(compBlock.component('bv'), Var)
        self.assertIsInstance(compBlock.component('c'), Constraint)
        self.assertIsInstance(compBlock.component('bc'), Constraint)

    def test_nl_descend_into_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.nl').apply_to(m)
        self.check_nl(m)

    def test_nl_on_disjunct(self):
        m = self.get_model()
        TransformationFactory('mpec.nl').apply_to(m.disjunct1)
        self.check_nl(m)