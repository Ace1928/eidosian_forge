from pyomo.core.expr.numvalue import value
from pyomo.solvers.plugins.solvers.cplex_direct import CPLEXDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.opt.base import SolverFactory
Add a column to the solver's model

        This will add the Pyomo variable var to the solver's
        model, and put the coefficients on the associated
        constraints in the solver model. If the obj_coef is
        not zero, it will add obj_coef*var to the objective
        of the solver's model.

        Parameters
        ----------
        var: Var (scalar Var or single _VarData)
        obj_coef: float
        constraints: list of solver constraints
        coefficients: list of coefficients to put on var in the associated constraint
        