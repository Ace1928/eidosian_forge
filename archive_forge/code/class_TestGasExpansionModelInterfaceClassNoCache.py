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
class TestGasExpansionModelInterfaceClassNoCache(unittest.TestCase):

    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface()
        constraints = list(model.component_data_objects(pyo.Constraint))
        variables = list(model.component_data_objects(pyo.Var))
        n_eqn = len(constraints)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        n_var = len(variables)
        matching = igraph.maximum_matching(variables, constraints)
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_var)
        self.assertEqual(len(values), n_var)
        self.assertIs(matching[model.ideal_gas[0]], model.P[0])

    def test_triangularize(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)]
        self.assertEqual(len(partition), N + 1)
        for i in model.streams:
            variables = ComponentSet([var for var, _ in partition[i]])
            constraints = ComponentSet([con for _, con in partition[i]])
            if i == model.streams.first():
                self.assertEqual(variables, ComponentSet([model.P[0]]))
            else:
                pred_vars = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
                pred_cons = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
                self.assertEqual(pred_vars, variables)
                self.assertEqual(pred_cons, constraints)

    def test_maps_from_triangularization(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N + 1)
        self.assertEqual(len(con_values), N + 1)
        self.assertEqual(var_block_map[model.P[0]], 0)
        for i in model.streams:
            if i != model.streams.first():
                self.assertEqual(var_block_map[model.rho[i]], i)
                self.assertEqual(var_block_map[model.T[i]], i)
                self.assertEqual(var_block_map[model.P[i]], i)
                self.assertEqual(var_block_map[model.F[i]], i)
                self.assertEqual(con_block_map[model.ideal_gas[i]], i)
                self.assertEqual(con_block_map[model.expansion[i]], i)
                self.assertEqual(con_block_map[model.mbal[i]], i)
                self.assertEqual(con_block_map[model.ebal[i]], i)

    def test_diagonal_blocks(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
        self.assertEqual(len(var_blocks), N + 1)
        self.assertEqual(len(con_blocks), N + 1)
        for i, (vars, cons) in enumerate(zip(var_blocks, con_blocks)):
            var_set = ComponentSet(vars)
            con_set = ComponentSet(cons)
            if i == 0:
                pred_var_set = ComponentSet([model.P[0]])
                self.assertEqual(pred_var_set, var_set)
                pred_con_set = ComponentSet([model.ideal_gas[0]])
                self.assertEqual(pred_con_set, con_set)
            else:
                pred_var_set = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
                pred_con_set = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
                self.assertEqual(pred_var_set, var_set)
                self.assertEqual(pred_con_set, con_set)

    def test_diagonal_blocks_with_cached_maps(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface()
        variables = []
        variables.extend(model.P.values())
        variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
        variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
        constraints = list(model.component_data_objects(pyo.Constraint))
        igraph.block_triangularize(variables, constraints)
        var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
        self.assertIs(igraph.row_block_map, None)
        self.assertIs(igraph.col_block_map, None)