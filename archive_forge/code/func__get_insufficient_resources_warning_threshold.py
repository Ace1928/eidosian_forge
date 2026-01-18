import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
@lru_cache()
def _get_insufficient_resources_warning_threshold() -> float:
    if _is_ray_cluster():
        return float(os.environ.get('TUNE_WARN_INSUFFICENT_RESOURCE_THRESHOLD_S_AUTOSCALER', '60'))
    else:
        return float(os.environ.get('TUNE_WARN_INSUFFICENT_RESOURCE_THRESHOLD_S', '60'))