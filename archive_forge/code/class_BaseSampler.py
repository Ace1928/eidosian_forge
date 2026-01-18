import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class BaseSampler(Sampler):

    def __str__(self):
        return 'Base'