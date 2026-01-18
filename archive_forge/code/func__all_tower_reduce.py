from collections import defaultdict
import numpy as np
import tree  # pip install dm_tree
from typing import Dict
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.typing import PolicyID
def _all_tower_reduce(path, *tower_data):
    """Reduces stats across towers based on their stats-dict paths."""
    if len(path) == 1 and path[0] == 'td_error':
        return np.concatenate(tower_data, axis=0)
    elif tower_data[0] is None:
        return None
    if isinstance(path[-1], str):
        if path[-1].startswith('min_'):
            return np.nanmin(tower_data)
        elif path[-1].startswith('max_'):
            return np.nanmax(tower_data)
    if np.isnan(tower_data).all():
        return np.nan
    return np.nanmean(tower_data)