import collections
import logging
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import GradInfoDict, LearnerStatsDict, ResultDict
def _extract_stats(stats: Dict, key: str) -> Dict[str, Any]:
    if key in stats:
        return stats[key]
    multiagent_stats = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            if key in v:
                multiagent_stats[k] = v[key]
    return multiagent_stats