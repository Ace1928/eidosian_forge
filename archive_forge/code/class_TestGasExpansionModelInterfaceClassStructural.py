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
class TestGasExpansionModelInterfaceClassStructural(unittest.TestCase):

    def test_imperfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)
        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)

    def test_perfect_matching(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)
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
        igraph = IncidenceGraphInterface(model)
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
        """
        This tests the maps from variables and constraints to their diagonal
        blocks returned by map_nodes_to_block_triangular_indices
        """
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)
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

    def test_triangularize_submatrix(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)
        variables = []
        half = N // 2
        variables.extend((model.P[i] for i in model.streams if i >= half))
        variables.extend((model.T[i] for i in model.streams if i > half))
        variables.extend((model.rho[i] for i in model.streams if i > half))
        variables.extend((model.F[i] for i in model.streams if i > half))
        constraints = []
        constraints.extend((model.ideal_gas[i] for i in model.streams if i >= half))
        constraints.extend((model.expansion[i] for i in model.streams if i > half))
        constraints.extend((model.mbal[i] for i in model.streams if i > half))
        constraints.extend((model.ebal[i] for i in model.streams if i > half))
        var_blocks, con_blocks = igraph.block_triangularize(variables, constraints)
        partition = [list(zip(vblock, cblock)) for vblock, cblock in zip(var_blocks, con_blocks)]
        self.assertEqual(len(partition), N - half + 1)
        for i in model.streams:
            idx = i - half
            variables = ComponentSet([var for var, _ in partition[idx]])
            constraints = ComponentSet([con for _, con in partition[idx]])
            if i == half:
                self.assertEqual(variables, ComponentSet([model.P[half]]))
            elif i > half:
                pred_var = ComponentSet([model.rho[i], model.T[i], model.P[i], model.F[i]])
                pred_con = ComponentSet([model.ideal_gas[i], model.expansion[i], model.mbal[i], model.ebal[i]])
                self.assertEqual(variables, pred_var)
                self.assertEqual(constraints, pred_con)

    def test_maps_from_triangularization_submatrix(self):
        N = 5
        model = make_gas_expansion_model(N)
        igraph = IncidenceGraphInterface(model)
        variables = []
        half = N // 2
        variables.extend((model.P[i] for i in model.streams if i >= half))
        variables.extend((model.T[i] for i in model.streams if i > half))
        variables.extend((model.rho[i] for i in model.streams if i > half))
        variables.extend((model.F[i] for i in model.streams if i > half))
        constraints = []
        constraints.extend((model.ideal_gas[i] for i in model.streams if i >= half))
        constraints.extend((model.expansion[i] for i in model.streams if i > half))
        constraints.extend((model.mbal[i] for i in model.streams if i > half))
        constraints.extend((model.ebal[i] for i in model.streams if i > half))
        var_block_map, con_block_map = igraph.map_nodes_to_block_triangular_indices(variables, constraints)
        var_values = set(var_block_map.values())
        con_values = set(con_block_map.values())
        self.assertEqual(len(var_values), N - half + 1)
        self.assertEqual(len(con_values), N - half + 1)
        self.assertEqual(var_block_map[model.P[half]], 0)
        for i in model.streams:
            if i > half:
                idx = i - half
                self.assertEqual(var_block_map[model.rho[i]], idx)
                self.assertEqual(var_block_map[model.T[i]], idx)
                self.assertEqual(var_block_map[model.P[i]], idx)
                self.assertEqual(var_block_map[model.F[i]], idx)
                self.assertEqual(con_block_map[model.ideal_gas[i]], idx)
                self.assertEqual(con_block_map[model.expansion[i]], idx)
                self.assertEqual(con_block_map[model.mbal[i]], idx)
                self.assertEqual(con_block_map[model.ebal[i]], idx)

    def test_exception(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)
        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.maximum_matching(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))
        with self.assertRaises(RuntimeError) as exc:
            variables = [model.P]
            constraints = [model.ideal_gas]
            igraph.block_triangularize(variables, constraints)
        self.assertIn('must be unindexed', str(exc.exception))

    @unittest.skipUnless(scipy_available, 'scipy is not available.')
    def test_remove(self):
        model = make_gas_expansion_model()
        igraph = IncidenceGraphInterface(model)
        n_eqn = len(list(model.component_data_objects(pyo.Constraint)))
        matching = igraph.maximum_matching()
        values = ComponentSet(matching.values())
        self.assertEqual(len(matching), n_eqn)
        self.assertEqual(len(values), n_eqn)
        variable_set = ComponentSet(igraph.variables)
        self.assertIn(model.F[0], variable_set)
        self.assertIn(model.F[2], variable_set)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        underconstrained_set = ComponentSet(var_dmp.unmatched + var_dmp.underconstrained)
        self.assertIn(model.F[0], underconstrained_set)
        self.assertIn(model.F[2], underconstrained_set)
        N, M = igraph.incidence_matrix.shape
        vars_to_remove = [model.F[0], model.F[2]]
        cons_to_remove = (model.mbal[1], model.mbal[2])
        igraph.remove_nodes(vars_to_remove, cons_to_remove)
        variable_set = ComponentSet(igraph.variables)
        self.assertNotIn(model.F[0], variable_set)
        self.assertNotIn(model.F[2], variable_set)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        underconstrained_set = ComponentSet(var_dmp.unmatched + var_dmp.underconstrained)
        self.assertNotIn(model.F[0], underconstrained_set)
        self.assertNotIn(model.F[2], underconstrained_set)
        N_new, M_new = igraph.incidence_matrix.shape
        self.assertEqual(N_new, N - len(cons_to_remove))
        self.assertEqual(M_new, M - len(vars_to_remove))