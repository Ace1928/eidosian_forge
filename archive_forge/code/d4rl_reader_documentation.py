import logging
import gymnasium as gym
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
from typing import Dict
Initializes a D4RLReader instance.

        Args:
            inputs: String corresponding to the D4RL environment name.
            ioctx: Current IO context object.
        