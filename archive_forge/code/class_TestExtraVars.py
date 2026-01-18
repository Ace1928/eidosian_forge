import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(networkx_available, 'networkx is not available.')
@unittest.skipUnless(scipy_available, 'scipy is not available.')
class TestExtraVars(unittest.TestCase):

    def test_unused_var(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var()
        m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
        igraph = IncidenceGraphInterface(m)
        self.assertEqual(igraph.incidence_matrix.shape, (1, 1))

    def test_reference(self):
        m = pyo.ConcreteModel()
        m.v1 = pyo.Var()
        m.ref = pyo.Reference(m.v1)
        m.c1 = pyo.Constraint(expr=m.v1 == 1.0)
        igraph = IncidenceGraphInterface(m)
        self.assertEqual(igraph.incidence_matrix.shape, (1, 1))