import numpy as np
from gymnasium.spaces import Box
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module import RLModule
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.models.base import STATE_OUT
An RLModule that always knows the current EpisodeID and EnvID and
    returns these in its actions.