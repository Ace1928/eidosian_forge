import logging
import re
import sys
import itertools
import operator
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.common.dependencies import attempt_import
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import is_fixed, value, minimize, maximize
from pyomo.core.base.suffix import Suffix
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt.base.solvers import OptSolver
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.common.collections import ComponentMap, ComponentSet, Bunch
from pyomo.opt import SolverFactory
from pyomo.core.kernel.conic import (
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
def _load_duals(self, objs_to_load=None):
    if not hasattr(self._pyomo_model, 'dual'):
        self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
    con_map = self._pyomo_con_to_solver_con_map
    reverse_con_map = self._solver_con_to_pyomo_con_map
    cone_map = self._pyomo_cone_to_solver_cone_map
    reverse_cone_map = self._solver_cone_to_pyomo_cone_map
    dual = self._pyomo_model.dual
    if objs_to_load is None:
        mosek_cons_to_load = range(self._solver_model.getnumcon())
        vals = [0.0] * len(mosek_cons_to_load)
        self._solver_model.gety(self._whichsol, vals)
        for mosek_con, val in zip(mosek_cons_to_load, vals):
            pyomo_con = reverse_con_map[mosek_con]
            dual[pyomo_con] = val
        'TODO wrong length, needs to be getnumvars()\n            # cones\n            mosek_cones_to_load = range(self._solver_model.getnumcone())\n            vals = [0.0]*len(mosek_cones_to_load)\n            self._solver_model.getsnx(self._whichsol, vals)\n            for mosek_cone, val in zip(mosek_cones_to_load, vals):\n                pyomo_cone = reverse_cone_map[mosek_cone]\n                dual[pyomo_cone] = val\n            UPDATE: the following code gets the dual info from cones,\n                    but each cones dual values are passed as lists\n            '
        if self._version[0] <= 9:
            vals = [0.0] * self._solver_model.getnumvar()
            self._solver_model.getsnx(self._whichsol, vals)
            for mosek_cone in range(self._solver_model.getnumcone()):
                dim = self._solver_model.getnumconemem(mosek_cone)
                members = [0] * dim
                self._solver_model.getcone(mosek_cone, members)
                pyomo_cone = reverse_cone_map[mosek_cone]
                dual[pyomo_cone] = tuple((vals[i] for i in members))
        else:
            mosek_cones_to_load = range(self._solver_model.getnumacc())
            mosek_cone_dims = [self._solver_model.getaccn(i) for i in mosek_cones_to_load]
            vals = self._solver_model.getaccdotys(self._whichsol)
            dim = 0
            for mosek_cone in mosek_cones_to_load:
                pyomo_cone = reverse_cone_map[mosek_cone]
                dual[pyomo_cone] = tuple(vals[dim:dim + mosek_cone_dims[mosek_cone]])
                dim += mosek_cone_dims[mosek_cone]
    else:
        mosek_cons_to_load = []
        mosek_cones_to_load = []
        for obj in objs_to_load:
            if obj in con_map:
                mosek_cons_to_load.append(con_map[obj])
            else:
                mosek_cones_to_load.append(cone_map[obj])
        if len(mosek_cons_to_load) > 0:
            mosek_cons_first = min(mosek_cons_to_load)
            mosek_cons_last = max(mosek_cons_to_load)
            vals = [0.0] * (mosek_cons_last - mosek_cons_first + 1)
            self._solver_model.getyslice(self._whichsol, mosek_cons_first, mosek_cons_last, vals)
            for mosek_con in mosek_cons_to_load:
                slice_index = mosek_con - mosek_cons_first
                val = vals[slice_index]
                pyomo_con = reverse_con_map[mosek_con]
                dual[pyomo_con] = val
            'TODO wrong length, needs to be getnumvars()\n            # cones\n            mosek_cones_first = min(mosek_cones_to_load)\n            mosek_cones_last = max(mosek_cones_to_load)\n            vals = [0.0]*(mosek_cones_last - mosek_cones_first + 1)\n            self._solver_model.getsnxslice(self._whichsol,\n                                           mosek_cones_first,\n                                           mosek_cones_last,\n                                           vals)\n            for mosek_cone in mosek_cones_to_load:\n                slice_index = mosek_cone - mosek_cones_first\n                val = vals[slice_index]\n                pyomo_cone = reverse_cone_map[mosek_cone]\n                dual[pyomo_cone] = val\n            '
        if len(mosek_cones_to_load) > 0:
            if self._version[0] <= 9:
                vals = [0] * self._solver_model.getnumvar()
                self._solver_model.getsnx(self._whichsol, vals)
                for mosek_cone in mosek_cones_to_load:
                    dim = self._solver_model.getnumconemem(mosek_cone)
                    members = [0] * dim
                    self._solver_model.getcone(mosek_cone, members)
                    pyomo_cone = reverse_cone_map[mosek_cone]
                    dual[pyomo_cone] = tuple((vals[i] for i in members))
            else:
                for mosek_cone in mosek_cones_to_load:
                    pyomo_cone = reverse_cone_map[mosek_cone]
                    dual[pyomo_cone] = tuple(self._solver_model.getaccdoty(self._whichsol, mosek_cone))