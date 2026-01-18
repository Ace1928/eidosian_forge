import copy
import logging
import math
from ray import cloudpickle
from typing import Dict, List, Optional, Union
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict
class _BOHBJobWrapper:
    """Mock object for HpBandSter to process."""

    def __init__(self, loss: float, budget: float, config: Dict):
        self.result = {'loss': loss}
        self.kwargs = {'budget': budget, 'config': config.copy()}
        self.exception = None