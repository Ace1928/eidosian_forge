import numpy as np
import pprint
from typing import Any, Mapping
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.annotations import DeveloperAPI
class _StringValue:

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value