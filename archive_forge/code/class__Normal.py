import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
class _Normal(Normal):

    def sample(self, domain: 'Float', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        assert not domain.lower or domain.lower == float('-inf'), 'Normal sampling does not allow a lower value bound.'
        assert not domain.upper or domain.upper == float('inf'), 'Normal sampling does not allow a upper value bound.'
        items = random_state.normal(self.mean, self.sd, size=size)
        return items if len(items) > 1 else domain.cast(items[0])