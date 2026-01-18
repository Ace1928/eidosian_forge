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
class TestIndexedBlock(unittest.TestCase):

    def test_block_data_obj(self):
        m = pyo.ConcreteModel()
        m.block = pyo.Block([1, 2, 3])
        m.block[1].subblock = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m.block[1])
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        self.assertEqual(len(var_dmp.unmatched), 1)
        self.assertEqual(len(con_dmp.unmatched), 1)
        msg = 'Unsupported type.*_BlockData'
        with self.assertRaisesRegex(TypeError, msg):
            igraph = IncidenceGraphInterface(m.block)