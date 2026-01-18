from typing import Any, Callable, Dict, List, Tuple
from unittest import TestCase
from triad import SerializableRLock
from tune import (
from tune._utils import assert_close
from tune.noniterative.objective import validate_noniterative_objective
def _generate_values(self, expr: StochasticExpression, obj: Callable[..., float], logger: Any=None) -> List[Any]:
    params = dict(a=expr)
    trial = Trial('x', params, metadata={})
    o = self.make_optimizer(max_iter=30)
    lock = SerializableRLock()
    values: List[Any] = []

    @noniterative_objective
    def objective(a: Any) -> float:
        with lock:
            values.append(a)
        return obj(a)
    o.run(objective, trial, logger=logger)
    return values