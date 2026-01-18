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
class TestDulmageMendelsohnInterface(unittest.TestCase):

    def test_degenerate_solid_phase_model(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        igraph = IncidenceGraphInterface()
        var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)
        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())
        self.assertEqual(len(var_dmp[0] + var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0] + var_dmp[1]:
            self.assertIn(var, underconstrained_vars)
        self.assertEqual(len(con_dmp[2]), len(underconstrained_cons))
        for con in con_dmp[2]:
            self.assertIn(con, underconstrained_cons)
        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)
        self.assertEqual(len(var_dmp[2]), len(overconstrained_vars))
        for var in var_dmp[2]:
            self.assertIn(var, overconstrained_vars)
        self.assertEqual(len(con_dmp[0] + con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0] + con_dmp[1]:
            self.assertIn(con, overconstrained_cons)

    def test_named_tuple(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        igraph = IncidenceGraphInterface()
        var_dmp, con_dmp = igraph.dulmage_mendelsohn(variables, constraints)
        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())
        dmp_vars_under = var_dmp.unmatched + var_dmp.underconstrained
        dmp_vars_over = var_dmp.overconstrained
        dmp_cons_under = con_dmp.underconstrained
        dmp_cons_over = con_dmp.unmatched + con_dmp.overconstrained
        self.assertEqual(len(dmp_vars_under), len(underconstrained_vars))
        for var in dmp_vars_under:
            self.assertIn(var, underconstrained_vars)
        self.assertEqual(len(dmp_cons_under), len(underconstrained_cons))
        for con in dmp_cons_under:
            self.assertIn(con, underconstrained_cons)
        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)
        self.assertEqual(len(dmp_vars_over), len(overconstrained_vars))
        for var in dmp_vars_over:
            self.assertIn(var, overconstrained_vars)
        self.assertEqual(len(dmp_cons_over), len(overconstrained_cons))
        for con in dmp_cons_over:
            self.assertIn(con, overconstrained_cons)

    @unittest.skipUnless(scipy_available, 'scipy is not available.')
    def test_incidence_graph(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        graph = get_incidence_graph(variables, constraints)
        matrix = get_structural_incidence_matrix(variables, constraints)
        from_matrix = from_biadjacency_matrix(matrix)
        self.assertEqual(graph.nodes, from_matrix.nodes)
        self.assertEqual(graph.edges, from_matrix.edges)

    def test_dm_graph_interface(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        graph = get_incidence_graph(variables, constraints)
        M, N = (len(constraints), len(variables))
        top_nodes = list(range(M))
        con_dmp, var_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)
        con_dmp = tuple(([constraints[i] for i in subset] for subset in con_dmp))
        var_dmp = tuple(([variables[i - M] for i in subset] for subset in var_dmp))
        underconstrained_vars = ComponentSet(m.flow_comp.values())
        underconstrained_vars.add(m.flow)
        underconstrained_cons = ComponentSet(m.flow_eqn.values())
        self.assertEqual(len(var_dmp[0] + var_dmp[1]), len(underconstrained_vars))
        for var in var_dmp[0] + var_dmp[1]:
            self.assertIn(var, underconstrained_vars)
        self.assertEqual(len(con_dmp[2]), len(underconstrained_cons))
        for con in con_dmp[2]:
            self.assertIn(con, underconstrained_cons)
        overconstrained_cons = ComponentSet(m.holdup_eqn.values())
        overconstrained_cons.add(m.density_eqn)
        overconstrained_cons.add(m.sum_eqn)
        overconstrained_vars = ComponentSet(m.x.values())
        overconstrained_vars.add(m.rho)
        self.assertEqual(len(var_dmp[2]), len(overconstrained_vars))
        for var in var_dmp[2]:
            self.assertIn(var, overconstrained_vars)
        self.assertEqual(len(con_dmp[0] + con_dmp[1]), len(overconstrained_cons))
        for con in con_dmp[0] + con_dmp[1]:
            self.assertIn(con, overconstrained_cons)

    @unittest.skipUnless(scipy_available, 'scipy is not available.')
    def test_remove(self):
        m = make_degenerate_solid_phase_model()
        variables = list(m.component_data_objects(pyo.Var))
        constraints = list(m.component_data_objects(pyo.Constraint))
        igraph = IncidenceGraphInterface(m)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        var_con_set = ComponentSet(igraph.variables + igraph.constraints)
        underconstrained_set = ComponentSet(var_dmp.unmatched + var_dmp.underconstrained)
        self.assertIn(m.flow_comp[1], var_con_set)
        self.assertIn(m.flow_eqn[1], var_con_set)
        self.assertIn(m.flow_comp[1], underconstrained_set)
        N, M = igraph.incidence_matrix.shape
        vars_to_remove = [m.flow_comp[1]]
        cons_to_remove = [m.flow_eqn[1]]
        igraph.remove_nodes(vars_to_remove + cons_to_remove)
        var_dmp, con_dmp = igraph.dulmage_mendelsohn()
        var_con_set = ComponentSet(igraph.variables + igraph.constraints)
        underconstrained_set = ComponentSet(var_dmp.unmatched + var_dmp.underconstrained)
        self.assertNotIn(m.flow_comp[1], var_con_set)
        self.assertNotIn(m.flow_eqn[1], var_con_set)
        self.assertNotIn(m.flow_comp[1], underconstrained_set)
        N_new, M_new = igraph.incidence_matrix.shape
        self.assertEqual(N_new, N - len(cons_to_remove))
        self.assertEqual(M_new, M - len(vars_to_remove))

    def test_recover_matching_from_dulmage_mendelsohn(self):
        m = make_degenerate_solid_phase_model()
        igraph = IncidenceGraphInterface(m)
        vdmp, cdmp = igraph.dulmage_mendelsohn()
        vmatch = vdmp.underconstrained + vdmp.square + vdmp.overconstrained
        cmatch = cdmp.underconstrained + cdmp.square + cdmp.overconstrained
        self.assertEqual(len(ComponentSet(vmatch)), len(vmatch))
        self.assertEqual(len(ComponentSet(cmatch)), len(cmatch))
        matching = list(zip(vmatch, cmatch))
        for var, con in matching:
            var_in_con = ComponentSet(igraph.get_adjacent_to(con))
            self.assertIn(var, var_in_con)