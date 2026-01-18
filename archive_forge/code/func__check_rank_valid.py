import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_rank_valid(g, rank: int):
    """Check the rank: 0 <= rank < world_size."""
    if rank < 0:
        raise ValueError("rank '{}' is negative.".format(rank))
    if rank >= g.world_size:
        raise ValueError("rank '{}' must be less than world size '{}'".format(rank, g.world_size))