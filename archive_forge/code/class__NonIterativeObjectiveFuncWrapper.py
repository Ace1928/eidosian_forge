import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check
from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
class _NonIterativeObjectiveFuncWrapper(NonIterativeObjectiveFunc):

    def __init__(self, min_better: bool):
        self._min_better = min_better

    @property
    def min_better(self) -> bool:
        return self._min_better

    def generate_sort_metric(self, value: float) -> float:
        return float(value) if self._min_better else -float(value)

    @no_type_check
    def run(self, trial: Trial) -> TrialReport:
        if self._orig_input:
            result = self._func(trial)
        else:
            result = self._func(**trial.params.simple_value, **trial.dfs)
        return self._output_f(result, trial)

    @no_type_check
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    @no_type_check
    @staticmethod
    def from_func(func: Callable, min_better: bool) -> '_NonIterativeObjectiveFuncWrapper':
        f = _NonIterativeObjectiveFuncWrapper(min_better=min_better)
        w = _NonIterativeObjectiveWrapper(func)
        f._func = w._func
        f._orig_input = w._orig_input
        f._output_f = w._rt.to_report
        return f