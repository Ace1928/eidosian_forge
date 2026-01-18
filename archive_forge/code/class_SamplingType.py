import copy
from enum import IntEnum
from functools import cached_property
from typing import Callable, List, Optional, Union
import torch
class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2
    BEAM = 3