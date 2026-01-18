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
def _add_cones(self, cones, num_cones):
    cone_names = tuple((self._symbol_map.getSymbol(c, self._labeler) for c in cones))
    if self._version[0] < 10:
        cone_num = self._solver_model.getnumcone()
        cone_indices = range(cone_num, cone_num + num_cones)
        cone_type, cone_param, cone_members = zip(*map(self._get_cone_data, cones))
        for i in range(num_cones):
            members = tuple((self._pyomo_var_to_solver_var_map[c_m] for c_m in cone_members[i]))
            self._solver_model.appendcone(cone_type[i], cone_param[i], members)
            self._solver_model.putconename(cone_indices[i], cone_names[i])
        self._pyomo_cone_to_solver_cone_map.update(zip(cones, cone_indices))
        self._solver_cone_to_pyomo_cone_map.update(zip(cone_indices, cones))
        for i, c in enumerate(cones):
            self._vars_referenced_by_con[c] = cone_members[i]
            for v in cone_members[i]:
                self._referenced_variables[v] += 1
    else:
        domain_dims, domain_indices, cone_members = zip(*map(self._get_acc_domain, cones))
        total_dim = sum(domain_dims)
        numafe = self._solver_model.getnumafe()
        numacc = self._solver_model.getnumacc()
        members = tuple((self._pyomo_var_to_solver_var_map[c_m] for c_m in itertools.chain(*cone_members)))
        afe_indices = tuple(range(numafe, numafe + total_dim))
        acc_indices = tuple(range(numacc, numacc + num_cones))
        self._solver_model.appendafes(total_dim)
        self._solver_model.putafefentrylist(afe_indices, members, [1] * total_dim)
        self._solver_model.appendaccsseq(domain_indices, total_dim, afe_indices[0], None)
        for name in cone_names:
            self._solver_model.putaccname(numacc, name)
        self._pyomo_cone_to_solver_cone_map.update(zip(cones, acc_indices))
        self._solver_cone_to_pyomo_cone_map.update(zip(acc_indices, cones))
        for i, c in enumerate(cones):
            self._vars_referenced_by_con[c] = cone_members[i]
            for v in cone_members[i]:
                self._referenced_variables[v] += 1