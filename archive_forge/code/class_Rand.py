import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class Rand(RandBase):
    """Continuous uniform random variables.
    Please read |SpaceTutorial|.

    :param low: range low bound (inclusive)
    :param high: range high bound (exclusive)
    :param q: step between adjacent values, if set, the value will be rounded
      using ``q``, defaults to None
    :param log: whether to do uniform sampling in log space, defaults to False.
      If True, ``low`` must be positive and lower values get higher chance to be sampled
    """

    def __init__(self, low: float, high: float, q: Optional[float]=None, log: bool=False, include_high: bool=True):
        if include_high:
            assert_or_throw(high >= low, ValueError(f'{high} < {low}'))
        else:
            assert_or_throw(high > low, ValueError(f'{high} <= {low}'))
        assert_or_throw(q is None or q > 0, ValueError(q))
        if log:
            assert_or_throw(low > 0.0, ValueError(f'for log sampling, low ({low}) must be greater than 0.0'))
        self.low = low
        self.high = high
        self.include_high = include_high
        super().__init__(q, log)

    @property
    def jsondict(self) -> Dict[str, Any]:
        res = dict(_expr_='rand', low=self.low, high=self.high, q=self.q, log=self.log, include_high=self.include_high)
        return res

    def generate(self, seed: Any=None) -> float:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.uniform()
        if self.q is None:
            return float(uniform_to_continuous(value, self.low, self.high, log=self.log))
        else:
            return float(uniform_to_discrete(value, self.low, self.high, q=self.q, log=self.log, include_high=self.include_high))

    def __repr__(self) -> str:
        return f'Rand(low={self.low}, high={self.high}, q={self.q}, log={self.log}, include_high={self.include_high})'