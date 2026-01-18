import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class LogUniform(Sampler):

    def __init__(self, base: float=10):
        self.base = base
        assert self.base > 0, 'Base has to be strictly greater than 0'

    def __str__(self):
        return 'LogUniform'