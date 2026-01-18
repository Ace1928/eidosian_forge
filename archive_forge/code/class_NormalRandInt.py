import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class NormalRandInt(RandBase):
    """Normally distributed random integer values.
    Please read |SpaceTutorial|.

    :param mu: mean of the normal distribution
    :param sigma: standard deviation of the normal distribution
    """

    def __init__(self, mu: int, sigma: float, q: int=1):
        assert_or_throw(sigma > 0, ValueError(sigma))
        assert_or_throw(q > 0, ValueError(q))
        self.mu = mu
        self.sigma = sigma
        self.q = q
        super().__init__(q)

    @property
    def jsondict(self) -> Dict[str, Any]:
        return dict(_expr_='randnint', mu=self.mu, sigma=self.sigma, q=self.q)

    def generate(self, seed: Any=None) -> int:
        if seed is not None:
            np.random.seed(seed)
        value = np.random.normal()
        return int(normal_to_integers(value, mean=self.mu, sigma=self.sigma, q=self.q))

    def __repr__(self) -> str:
        return f'NormalRandInt(mu={self.mu}, sigma={self.sigma}, q={self.q})'