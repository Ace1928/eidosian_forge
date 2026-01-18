import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
@lru_cache()
def _get_cluster_resources_no_autoscaler() -> Dict:
    return ray.cluster_resources()