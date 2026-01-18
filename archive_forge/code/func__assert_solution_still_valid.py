import abc
from typing import Sequence, Dict, Optional, Mapping, NoReturn
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr import value
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.sol_reader import SolFileData
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.core.expr.visitor import replace_expressions
def _assert_solution_still_valid(self):
    if not self._valid:
        raise RuntimeError('The results in the solver are no longer valid.')