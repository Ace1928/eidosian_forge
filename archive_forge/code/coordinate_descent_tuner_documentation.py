import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config

        Check if candidate_config is better than best_config.

        Return a touple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        