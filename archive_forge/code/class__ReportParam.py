import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check
from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
class _ReportParam(AnnotatedParam):

    def to_report(self, v: Any, trial: Trial) -> TrialReport:
        raise NotImplementedError