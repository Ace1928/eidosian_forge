from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import logging
@declare_custom_block(name='BendersCutGenerator')
class BendersCutGeneratorData(_BlockData):

    def __init__(self, component):
        if not mpi4py_available:
            raise ImportError('BendersCutGenerator requires mpi4py.')
        if not numpy_available:
            raise ImportError('BendersCutGenerator requires numpy.')
        _BlockData.__init__(self, component)
        self.num_subproblems_by_rank = 0
        self.subproblems = list()
        self.complicating_vars_maps = list()
        self.root_vars = list()
        self.root_vars_indices = pyo.ComponentMap()
        self.root_etas = list()
        self.cuts = None
        self.subproblem_solvers = list()
        self.tol = None
        self.all_root_etas = list()
        self._subproblem_ndx_map = dict()

    def global_num_subproblems(self):
        return int(self.num_subproblems_by_rank.sum())

    def local_num_subproblems(self):
        return len(self.subproblems)

    def set_input(self, root_vars, tol=1e-06, comm=None):
        """
        It is very important for root_vars to be in the same order for every process.

        Parameters
        ----------
        root_vars
        tol
        """
        self.comm = None
        if comm is not None:
            self.comm = comm
        else:
            self.comm = MPI.COMM_WORLD
        self.num_subproblems_by_rank = np.zeros(self.comm.Get_size())
        del self.cuts
        self.cuts = pyo.ConstraintList()
        self.subproblems = list()
        self.root_etas = list()
        self.complicating_vars_maps = list()
        self.root_vars = list(root_vars)
        self.root_vars_indices = pyo.ComponentMap()
        for i, v in enumerate(self.root_vars):
            self.root_vars_indices[v] = i
        self.tol = tol
        self.subproblem_solvers = list()
        self.all_root_etas = list()
        self._subproblem_ndx_map = dict()

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, root_eta, subproblem_solver='gurobi_persistent', relax_subproblem_cons=False):
        _rank = np.argmin(self.num_subproblems_by_rank)
        self.num_subproblems_by_rank[_rank] += 1
        self.all_root_etas.append(root_eta)
        if _rank == self.comm.Get_rank():
            self.root_etas.append(root_eta)
            subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
            self.subproblems.append(subproblem)
            self.complicating_vars_maps.append(complicating_vars_map)
            _setup_subproblem(subproblem, root_vars=[complicating_vars_map[i] for i in self.root_vars if i in complicating_vars_map], relax_subproblem_cons=relax_subproblem_cons)
            self._subproblem_ndx_map[len(self.subproblems) - 1] = self.global_num_subproblems() - 1
            if isinstance(subproblem_solver, str):
                subproblem_solver = pyo.SolverFactory(subproblem_solver)
            self.subproblem_solvers.append(subproblem_solver)
            if isinstance(subproblem_solver, PersistentSolver):
                subproblem_solver.set_instance(subproblem)

    def generate_cut(self):
        coefficients = np.zeros(self.global_num_subproblems() * len(self.root_vars), dtype='d')
        constants = np.zeros(self.global_num_subproblems(), dtype='d')
        eta_coeffs = np.zeros(self.global_num_subproblems(), dtype='d')
        for local_subproblem_ndx in range(len(self.subproblems)):
            subproblem = self.subproblems[local_subproblem_ndx]
            global_subproblem_ndx = self._subproblem_ndx_map[local_subproblem_ndx]
            complicating_vars_map = self.complicating_vars_maps[local_subproblem_ndx]
            root_eta = self.root_etas[local_subproblem_ndx]
            coeff_ndx = global_subproblem_ndx * len(self.root_vars)
            subproblem.fix_complicating_vars = pyo.ConstraintList()
            var_to_con_map = pyo.ComponentMap()
            for root_var in self.root_vars:
                if root_var in complicating_vars_map:
                    sub_var = complicating_vars_map[root_var]
                    sub_var.set_value(root_var.value, skip_validation=True)
                    new_con = subproblem.fix_complicating_vars.add(sub_var - root_var.value == 0)
                    var_to_con_map[root_var] = new_con
            subproblem.fix_eta = pyo.Constraint(expr=subproblem._eta - root_eta.value == 0)
            subproblem._eta.set_value(root_eta.value, skip_validation=True)
            subproblem_solver = self.subproblem_solvers[local_subproblem_ndx]
            if subproblem_solver.name not in solver_dual_sign_convention:
                raise NotImplementedError('BendersCutGenerator is unaware of the dual sign convention of subproblem solver ' + subproblem_solver.name)
            sign_convention = solver_dual_sign_convention[subproblem_solver.name]
            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.add_constraint(c)
                subproblem_solver.add_constraint(subproblem.fix_eta)
                res = subproblem_solver.solve(tee=False, load_solutions=False, save_results=False)
                if res.solver.termination_condition != pyo.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem_solver.load_vars()
                subproblem_solver.load_duals()
            else:
                res = subproblem_solver.solve(subproblem, tee=False, load_solutions=False)
                if res.solver.termination_condition != pyo.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem.solutions.load_from(res)
            constants[global_subproblem_ndx] = pyo.value(subproblem._z)
            eta_coeffs[global_subproblem_ndx] = sign_convention * pyo.value(subproblem.dual[subproblem.obj_con])
            for root_var in self.root_vars:
                if root_var in complicating_vars_map:
                    c = var_to_con_map[root_var]
                    coefficients[coeff_ndx] = sign_convention * pyo.value(subproblem.dual[c])
                coeff_ndx += 1
            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.remove_constraint(c)
                subproblem_solver.remove_constraint(subproblem.fix_eta)
            del subproblem.fix_complicating_vars
            del subproblem.fix_eta
        total_num_subproblems = self.global_num_subproblems()
        global_constants = np.zeros(total_num_subproblems, dtype='d')
        global_coeffs = np.zeros(total_num_subproblems * len(self.root_vars), dtype='d')
        global_eta_coeffs = np.zeros(total_num_subproblems, dtype='d')
        comm = self.comm
        comm.Allreduce([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        comm.Allreduce([eta_coeffs, MPI.DOUBLE], [global_eta_coeffs, MPI.DOUBLE])
        comm.Allreduce([coefficients, MPI.DOUBLE], [global_coeffs, MPI.DOUBLE])
        global_constants = [float(i) for i in global_constants]
        global_coeffs = [float(i) for i in global_coeffs]
        global_eta_coeffs = [float(i) for i in global_eta_coeffs]
        coeff_ndx = 0
        cuts_added = list()
        for global_subproblem_ndx in range(total_num_subproblems):
            cut_expr = global_constants[global_subproblem_ndx]
            if cut_expr > self.tol:
                root_eta = self.all_root_etas[global_subproblem_ndx]
                cut_expr -= global_eta_coeffs[global_subproblem_ndx] * (root_eta - root_eta.value)
                for root_var in self.root_vars:
                    coeff = global_coeffs[coeff_ndx]
                    cut_expr -= coeff * (root_var - root_var.value)
                    coeff_ndx += 1
                new_cut = self.cuts.add(cut_expr <= 0)
                cuts_added.append(new_cut)
            else:
                coeff_ndx += len(self.root_vars)
        return cuts_added