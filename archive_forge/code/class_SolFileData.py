from typing import Tuple, Dict, Any, List
import io
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
class SolFileData:

    def __init__(self) -> None:
        self.primals: List[float] = list()
        self.duals: List[float] = list()
        self.var_suffixes: Dict[str, Dict[int, Any]] = dict()
        self.con_suffixes: Dict[str, Dict[Any]] = dict()
        self.obj_suffixes: Dict[str, Dict[int, Any]] = dict()
        self.problem_suffixes: Dict[str, List[Any]] = dict()
        self.other: List(str) = list()