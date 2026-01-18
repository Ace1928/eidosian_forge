import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning_fabric.utilities.rank_zero import _get_rank, rank_prefixed_message, rank_zero_only, rank_zero_warn
def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict['torch'])
    if 'torch.cuda' in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict['torch.cuda'])
    np.random.set_state(rng_state_dict['numpy'])
    version, state, gauss = rng_state_dict['python']
    python_set_rng_state((version, tuple(state), gauss))