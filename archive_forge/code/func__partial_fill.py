import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def _partial_fill(self, idx: List[int], params_list: Iterable[List[Any]]) -> Iterable['TuningParametersTemplate']:
    new_units = [u for i, u in enumerate(self._units) if i not in idx]
    has_grid = any((isinstance(x.expr, Grid) for x in new_units))
    has_stochastic = any((isinstance(x.expr, StochasticExpression) for x in new_units))
    for params in params_list:
        new_template = deepcopy(self._template)
        for pi, i in enumerate(idx):
            for path in self._units[i].positions:
                self._fill_path(new_template, path, params[pi])
        t = TuningParametersTemplate({})
        t._units = new_units
        t._template = new_template
        t._has_grid = has_grid
        t._has_stochastic = has_stochastic
        t._func_positions = self._func_positions
        if t.empty and len(t._func_positions) > 0:
            t._template = t._fill_funcs(t._template)
            t._func_positions = []
        yield t