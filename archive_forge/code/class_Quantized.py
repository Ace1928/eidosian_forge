import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class Quantized(Sampler):

    def __init__(self, sampler: Sampler, q: Union[float, int]):
        self.sampler = sampler
        self.q = q
        assert self.sampler, 'Quantized() expects a sampler instance'

    def get_sampler(self):
        return self.sampler

    def sample(self, domain: Domain, config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        if self.q == 1:
            return self.sampler.sample(domain, config, size, random_state=random_state)
        quantized_domain = copy(domain)
        quantized_domain.lower = np.ceil(domain.lower / self.q) * self.q
        quantized_domain.upper = np.floor(domain.upper / self.q) * self.q
        values = self.sampler.sample(quantized_domain, config, size, random_state=random_state)
        quantized = np.round(np.divide(values, self.q)) * self.q
        if not isinstance(quantized, np.ndarray):
            return domain.cast(quantized)
        return list(quantized)