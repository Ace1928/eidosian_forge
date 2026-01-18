import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
class _LogUniform(LogUniform):

    def sample(self, domain: 'Integer', config: Optional[Union[List[Dict], Dict]]=None, size: int=1, random_state: 'RandomState'=None):
        if not isinstance(random_state, _BackwardsCompatibleNumpyRng):
            random_state = _BackwardsCompatibleNumpyRng(random_state)
        assert domain.lower > 0, 'LogUniform needs a lower bound greater than 0'
        assert 0 < domain.upper < float('inf'), 'LogUniform needs a upper bound greater than 0'
        logmin = np.log(domain.lower) / np.log(self.base)
        logmax = np.log(domain.upper) / np.log(self.base)
        items = self.base ** random_state.uniform(logmin, logmax, size=size)
        items = np.floor(items).astype(int)
        return items if len(items) > 1 else domain.cast(items[0])