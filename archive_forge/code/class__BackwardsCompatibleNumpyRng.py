import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
class _BackwardsCompatibleNumpyRng:
    """Thin wrapper to ensure backwards compatibility between
    new and old numpy randomness generators.
    """
    _rng = None

    def __init__(self, generator_or_seed: Optional[Union['np_random_generator', np.random.RandomState, int]]=None):
        if generator_or_seed is None or isinstance(generator_or_seed, (np.random.RandomState, np_random_generator)):
            self._rng = generator_or_seed
        elif LEGACY_RNG:
            self._rng = np.random.RandomState(generator_or_seed)
        else:
            self._rng = np.random.default_rng(generator_or_seed)

    @property
    def legacy_rng(self) -> bool:
        return not isinstance(self._rng, np_random_generator)

    @property
    def rng(self):
        return self._rng if self._rng is not None else np.random

    def __getattr__(self, name: str) -> Any:
        if self.legacy_rng:
            if name == 'integers':
                name = 'randint'
            elif name == 'random':
                name = 'rand'
        return getattr(self.rng, name)