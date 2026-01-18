import os
import subprocess
import datetime
import io
from typing import Mapping, Optional, Sequence
from pyomo.common import Executable
from pyomo.common.config import ConfigValue, document_kwargs_from_configdict, ConfigDict
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.nl_writer import NLWriter, NLWriterInfo
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.config import SolverConfig
from pyomo.contrib.solver.factory import SolverFactory
from pyomo.contrib.solver.results import Results, TerminationCondition, SolutionStatus
from pyomo.contrib.solver.sol_reader import parse_sol_file
from pyomo.contrib.solver.solution import SolSolutionLoader
from pyomo.common.tee import TeeStream
from pyomo.core.expr.visitor import replace_expressions
from pyomo.core.expr.numvalue import value
from pyomo.core.base.suffix import Suffix
from pyomo.common.collections import ComponentMap
import logging
class IpoptSolutionLoader(SolSolutionLoader):

    def get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        if self._nl_info is None:
            raise RuntimeError('Solution loader does not currently have a valid solution. Please check results.TerminationCondition and/or results.SolutionStatus.')
        if len(self._nl_info.eliminated_vars) > 0:
            raise NotImplementedError('For now, turn presolve off (opt.config.writer_config.linear_presolve=False) to get dual variable values.')
        if self._sol_data is None:
            raise DeveloperError('Solution data is empty. This should not have happened. Report this error to the Pyomo Developers.')
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.variables)
            obj_scale = 1
        else:
            scale_list = self._nl_info.scaling.variables
            obj_scale = self._nl_info.scaling.objectives[0]
        sol_data = self._sol_data
        nl_info = self._nl_info
        zl_map = sol_data.var_suffixes['ipopt_zL_out']
        zu_map = sol_data.var_suffixes['ipopt_zU_out']
        rc = dict()
        for ndx, v in enumerate(nl_info.variables):
            scale = scale_list[ndx]
            v_id = id(v)
            rc[v_id] = (v, 0)
            if ndx in zl_map:
                zl = zl_map[ndx] * scale / obj_scale
                if abs(zl) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zl)
            if ndx in zu_map:
                zu = zu_map[ndx] * scale / obj_scale
                if abs(zu) > abs(rc[v_id][1]):
                    rc[v_id] = (v, zu)
        if vars_to_load is None:
            res = ComponentMap(rc.values())
            for v, _ in nl_info.eliminated_vars:
                res[v] = 0
        else:
            res = ComponentMap()
            for v in vars_to_load:
                if id(v) in rc:
                    res[v] = rc[id(v)][1]
                else:
                    res[v] = 0
        return res