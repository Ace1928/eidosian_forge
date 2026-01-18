import itertools
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
import numpy as np
import scipy.sparse as sps
class ExternalPyomoModel(ExternalGreyBoxModel):
    """
    This is an ExternalGreyBoxModel used to create an external model
    from existing Pyomo components. Given a system of variables and
    equations partitioned into "input" and "external" variables and
    "residual" and "external" equations, this class computes the
    residual of the "residual equations," as well as their Jacobian
    and Hessian, as a function of only the inputs.

    Pyomo components:
        f(x, y) == 0 # "Residual equations"
        g(x, y) == 0 # "External equations", dim(g) == dim(y)

    Effective constraint seen by this "external model":
        F(x) == f(x, y(x)) == 0
        where y(x) solves g(x, y) == 0

    """

    def __init__(self, input_vars, external_vars, residual_cons, external_cons, solver_class=None, solver_options=None, timer=None):
        """
        Arguments:
        ----------
        input_vars: list
            List of variables sent to this system by the outer solver
        external_vars: list
            List of variables that are solved for internally by this system
        residual_cons: list
            List of equality constraints whose residuals are exposed to
            the outer solver
        external_cons: list
            List of equality constraints used to solve for the external
            variables
        solver_class: Subclass of ImplicitFunctionSolver
            The solver object that is used to converge the system of
            equations defining the implicit function.
        solver_options: dict
            Options dict for the ImplicitFunctionSolver
        timer: HierarchicalTimer
            HierarchicalTimer object to which new timing categories introduced
            will be attached. If None, a new timer will be created.

        """
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer
        if solver_class is None:
            solver_class = SccImplicitFunctionSolver
        self._solver_class = solver_class
        if solver_options is None:
            solver_options = {}
        self._timer.start('__init__')
        self._block = create_subsystem_block(residual_cons + external_cons, input_vars + external_vars)
        self._block._obj = Objective(expr=0.0)
        self._timer.start('PyomoNLP')
        self._nlp = PyomoNLP(self._block)
        self._timer.stop('PyomoNLP')
        self._solver = self._solver_class(external_vars, external_cons, input_vars, timer=self._timer, **solver_options)
        assert len(external_vars) == len(external_cons)
        self.input_vars = input_vars
        self.external_vars = external_vars
        self.residual_cons = residual_cons
        self.external_cons = external_cons
        self.residual_con_multipliers = [None for _ in residual_cons]
        self.residual_scaling_factors = None
        self._input_output_coords = self._nlp.get_primal_indices(input_vars + external_vars)
        self._timer.stop('__init__')

    def n_inputs(self):
        return len(self.input_vars)

    def n_equality_constraints(self):
        return len(self.residual_cons)

    def input_names(self):
        return ['input_%i' % i for i in range(self.n_inputs())]

    def equality_constraint_names(self):
        return ['residual_%i' % i for i in range(self.n_equality_constraints())]

    def set_input_values(self, input_values):
        self._timer.start('set_inputs')
        solver = self._solver
        external_cons = self.external_cons
        external_vars = self.external_vars
        input_vars = self.input_vars
        solver.set_parameters(input_values)
        outputs = solver.evaluate_outputs()
        solver.update_pyomo_model()
        primals = self._nlp.get_primals()
        values = np.concatenate((input_values, outputs))
        primals[self._input_output_coords] = values
        self._nlp.set_primals(primals)
        self._timer.stop('set_inputs')

    def set_equality_constraint_multipliers(self, eq_con_multipliers):
        """
        Sets multipliers for residual equality constraints seen by the
        outer solver.

        """
        for i, val in enumerate(eq_con_multipliers):
            self.residual_con_multipliers[i] = val

    def set_external_constraint_multipliers(self, eq_con_multipliers):
        eq_con_multipliers = np.array(eq_con_multipliers)
        external_multipliers = self.calculate_external_constraint_multipliers(eq_con_multipliers)
        multipliers = np.concatenate((eq_con_multipliers, external_multipliers))
        cons = self.residual_cons + self.external_cons
        n_con = len(cons)
        assert n_con == self._nlp.n_constraints()
        duals = np.zeros(n_con)
        indices = self._nlp.get_constraint_indices(cons)
        duals[indices] = multipliers
        self._nlp.set_duals(duals)

    def calculate_external_constraint_multipliers(self, resid_multipliers):
        """
        Calculates the multipliers of the external constraints from the
        multipliers of the residual constraints (which are provided by
        the "outer" solver).

        """
        nlp = self._nlp
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_t = jgy.transpose()
        jfy_t = jfy.transpose()
        dfdg = -sps.linalg.splu(jgy_t.tocsc()).solve(jfy_t.toarray())
        resid_multipliers = np.array(resid_multipliers)
        external_multipliers = dfdg.dot(resid_multipliers)
        return external_multipliers

    def get_full_space_lagrangian_hessians(self):
        """
        Calculates terms of Hessian of full-space Lagrangian due to
        external and residual constraints. Note that multipliers are
        set by set_equality_constraint_multipliers. These matrices
        are used to calculate the Hessian of the reduced-space
        Lagrangian.

        """
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        hlxx = nlp.extract_submatrix_hessian_lag(x, x)
        hlxy = nlp.extract_submatrix_hessian_lag(x, y)
        hlyy = nlp.extract_submatrix_hessian_lag(y, y)
        return (hlxx, hlxy, hlyy)

    def calculate_reduced_hessian_lagrangian(self, hlxx, hlxy, hlyy):
        """
        Performs the matrix multiplications necessary to get the
        reduced space Hessian-of-Lagrangian term from the full-space
        terms.

        """
        hlxx = hlxx.toarray()
        hlxy = hlxy.toarray()
        hlyy = hlyy.toarray()
        dydx = self.evaluate_jacobian_external_variables()
        term1 = hlxx
        prod = hlxy.dot(dydx)
        term2 = prod + prod.transpose()
        term3 = hlyy.dot(dydx).transpose().dot(dydx)
        hess_lag = term1 + term2 + term3
        return hess_lag

    def evaluate_equality_constraints(self):
        return self._nlp.extract_subvector_constraints(self.residual_cons)

    def evaluate_jacobian_equality_constraints(self):
        self._timer.start('jacobian')
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        nf = len(f)
        nx = len(x)
        n_entries = nf * nx
        dydx = -1 * sps.linalg.splu(jgy.tocsc()).solve(jgx.toarray())
        dfdx = jfx + jfy.dot(dydx)
        full_sparse = _dense_to_full_sparse(dfdx)
        self._timer.stop('jacobian')
        return full_sparse

    def evaluate_jacobian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        dydx = -1 * sps.linalg.splu(jgy_csc).solve(jgx.toarray())
        return dydx

    def evaluate_hessian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        jgy_fact = sps.linalg.splu(jgy_csc)
        dydx = -1 * jgy_fact.solve(jgx.toarray())
        ny = len(y)
        nx = len(x)
        hgxx = np.array([get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in g])
        hgxy = np.array([get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in g])
        hgyy = np.array([get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in g])
        term1 = hgxx
        prod = hgxy.dot(dydx)
        term2 = prod + prod.transpose((0, 2, 1))
        term3 = hgyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)
        rhs = term1 + term2 + term3
        rhs.shape = (ny, nx * nx)
        sol = jgy_fact.solve(rhs)
        sol.shape = (ny, nx, nx)
        d2ydx2 = -sol
        return d2ydx2

    def evaluate_hessians_of_residuals(self):
        """
        This method computes the Hessian matrix of each equality
        constraint individually, rather than the sum of Hessians
        times multipliers.
        """
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)
        dydx = self.evaluate_jacobian_external_variables()
        ny = len(y)
        nf = len(f)
        nx = len(x)
        hfxx = np.array([get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in f])
        hfxy = np.array([get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in f])
        hfyy = np.array([get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in f])
        d2ydx2 = self.evaluate_hessian_external_variables()
        term1 = hfxx
        prod = hfxy.dot(dydx)
        term2 = prod + prod.transpose((0, 2, 1))
        term3 = hfyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)
        d2ydx2.shape = (ny, nx * nx)
        term4 = jfy.dot(d2ydx2)
        term4.shape = (nf, nx, nx)
        d2fdx2 = term1 + term2 + term3 + term4
        return d2fdx2

    def evaluate_hessian_equality_constraints(self):
        """
        This method actually evaluates the sum of Hessians times
        multipliers, i.e. the term in the Hessian of the Lagrangian
        due to these equality constraints.

        """
        self._timer.start('hessian')
        eq_con_multipliers = self.residual_con_multipliers
        self.set_external_constraint_multipliers(eq_con_multipliers)
        hlxx, hlxy, hlyy = self.get_full_space_lagrangian_hessians()
        hess_lag = self.calculate_reduced_hessian_lagrangian(hlxx, hlxy, hlyy)
        sparse = _dense_to_full_sparse(hess_lag)
        lower_triangle = sps.tril(sparse)
        self._timer.stop('hessian')
        return lower_triangle

    def set_equality_constraint_scaling_factors(self, scaling_factors):
        """
        Set scaling factors for the equality constraints that are exposed
        to a solver. These are the "residual equations" in this class.
        """
        self.residual_scaling_factors = np.array(scaling_factors)

    def get_equality_constraint_scaling_factors(self):
        """
        Get scaling factors for the equality constraints that are exposed
        to a solver. These are the "residual equations" in this class.
        """
        return self.residual_scaling_factors