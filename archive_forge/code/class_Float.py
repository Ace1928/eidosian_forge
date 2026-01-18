import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class Float(Domain):

    class _Uniform(Uniform):

        def sample(self, domain: 'Float', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
            if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
                random_state = _BackwardsCompatibleNumpyRng(random_state)
            assert domain.lower > float('-inf'), 'Uniform needs a lower bound'
            assert domain.upper < float('inf'), 'Uniform needs a upper bound'
            items = random_state.uniform(domain.lower, domain.upper, size=size)
            return items if len(items) > 1 else domain.cast(items[0])

    class _LogUniform(LogUniform):

        def sample(self, domain: 'Float', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
            if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
                random_state = _BackwardsCompatibleNumpyRng(random_state)
            assert domain.lower > 0, 'LogUniform needs a lower bound greater than 0'
            assert 0 < domain.upper < float('inf'), 'LogUniform needs a upper bound greater than 0'
            logmin = np.log(domain.lower) / np.log(self.base)
            logmax = np.log(domain.upper) / np.log(self.base)
            items = self.base ** random_state.uniform(logmin, logmax, size=size)
            return items if len(items) > 1 else domain.cast(items[0])

    class _Normal(Normal):

        def sample(self, domain: 'Float', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
            if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
                random_state = _BackwardsCompatibleNumpyRng(random_state)
            assert not domain.lower or domain.lower == float('-inf'), 'Normal sampling does not allow a lower value bound.'
            assert not domain.upper or domain.upper == float('inf'), 'Normal sampling does not allow a upper value bound.'
            items = random_state.normal(self.mean, self.sd, size=size)
            return items if len(items) > 1 else domain.cast(items[0])
    default_sampler_cls = _Uniform

    def __init__(self, lower: Optional[float], upper: Optional[float]):
        self.lower = lower if lower is not None else float('-inf')
        self.upper = upper if upper is not None else float('inf')

    def cast(self, value):
        return float(value)

    def uniform(self):
        if not self.lower > float('-inf'):
            raise ValueError('Uniform requires a lower bound. Make sure to set the `lower` parameter of `Float()`.')
        if not self.upper < float('inf'):
            raise ValueError('Uniform requires a upper bound. Make sure to set the `upper` parameter of `Float()`.')
        new = copy(self)
        new.set_sampler(self._Uniform())
        return new

    def loguniform(self, base: float=10):
        if not self.lower > 0:
            raise ValueError(f'LogUniform requires a lower bound greater than 0.Got: {self.lower}. Did you pass a variable that has been log-transformed? If so, pass the non-transformed value instead.')
        if not 0 < self.upper < float('inf'):
            raise ValueError(f'LogUniform requires a upper bound greater than 0. Got: {self.lower}. Did you pass a variable that has been log-transformed? If so, pass the non-transformed value instead.')
        new = copy(self)
        new.set_sampler(self._LogUniform(base))
        return new

    def normal(self, mean=0.0, sd=1.0):
        new = copy(self)
        new.set_sampler(self._Normal(mean, sd))
        return new

    def quantized(self, q: float):
        if self.lower > float('-inf') and (not isclose(self.lower / q, round(self.lower / q))):
            raise ValueError(f'Your lower variable bound {self.lower} is not divisible by quantization factor {q}.')
        if self.upper < float('inf') and (not isclose(self.upper / q, round(self.upper / q))):
            raise ValueError(f'Your upper variable bound {self.upper} is not divisible by quantization factor {q}.')
        new = copy(self)
        new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
        return new

    def is_valid(self, value: float):
        return self.lower <= value <= self.upper

    @property
    def domain_str(self):
        return f'({self.lower}, {self.upper})'