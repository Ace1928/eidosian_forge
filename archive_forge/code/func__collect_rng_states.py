import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning_fabric.utilities.rank_zero import _get_rank, rank_prefixed_message, rank_zero_only, rank_zero_warn
def _collect_rng_states(include_cuda: bool=True) -> Dict[str, Any]:
    """Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': python_get_rng_state()}
    if include_cuda:
        states['torch.cuda'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states