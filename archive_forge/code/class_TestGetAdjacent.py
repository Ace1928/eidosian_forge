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
class TestGetAdjacent(unittest.TestCase):

    def test_get_adjacent_to_var(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        adj_cons = igraph.get_adjacent_to(m.rho)
        self.assertEqual(ComponentSet(adj_cons), ComponentSet([m.holdup_eqn[1], m.holdup_eqn[2], m.holdup_eqn[3], m.density_eqn]))

    def test_get_adjacent_to_con(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        adj_vars = igraph.get_adjacent_to(m.density_eqn)
        self.assertEqual(ComponentSet(adj_vars), ComponentSet([m.x[1], m.x[2], m.x[3], m.rho]))

    def test_get_adjacent_exceptions(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface()
        msg = 'Cannot get components adjacent to'
        with self.assertRaisesRegex(RuntimeError, msg):
            adj_vars = igraph.get_adjacent_to(m.density_eqn)
        m.x[1].fix()
        igraph = IncidenceGraphInterface(m, include_fixed=False)
        msg = 'Cannot find component'
        with self.assertRaisesRegex(RuntimeError, msg):
            adj_cons = igraph.get_adjacent_to(m.x[1])