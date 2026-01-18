import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class Domain:
    """Base class to specify a type and valid range to sample parameters from.

    This base class is implemented by parameter spaces, like float ranges
    (``Float``), integer ranges (``Integer``), or categorical variables
    (``Categorical``). The ``Domain`` object contains information about
    valid values (e.g. minimum and maximum values), and exposes methods that
    allow specification of specific samplers (e.g. ``uniform()`` or
    ``loguniform()``).

    """
    sampler = None
    default_sampler_cls = None

    def cast(self, value):
        """Cast value to domain type"""
        return value

    def set_sampler(self, sampler, allow_override=False):
        if self.sampler and (not allow_override):
            raise ValueError('You can only choose one sampler for parameter domains. Existing sampler for parameter {}: {}. Tried to add {}'.format(self.__class__.__name__, self.sampler, sampler))
        self.sampler = sampler

    def get_sampler(self):
        sampler = self.sampler
        if not sampler:
            sampler = self.default_sampler_cls()
        return sampler

    def sample(self, config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        sampler = self.get_sampler()
        return sampler.sample(self, config=config, size=size, random_state=random_state)

    def is_grid(self):
        return isinstance(self.sampler, Grid)

    def is_function(self):
        return False

    def is_valid(self, value: Any):
        """Returns True if `value` is a valid value in this domain."""
        raise NotImplementedError

    @property
    def domain_str(self):
        return '(unknown)'