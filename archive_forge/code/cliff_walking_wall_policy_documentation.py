import gymnasium as gym
from typing import Dict, Union, List, Tuple, Optional
import numpy as np
from ray.rllib.policy.policy import Policy, ViewRequirement
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorStructType, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.debug import update_global_seed_if_necessary
Optimal RLlib policy for the CliffWalkingWallEnv environment, defined in
    ray/rllib/examples/env/cliff_walking_wall_env.py, with epsilon-greedy exploration.

    The policy takes a random action with probability epsilon, specified
    by `config["epsilon"]`, and the optimal action with probability  1 - epsilon.
    