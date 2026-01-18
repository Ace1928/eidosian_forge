import logging
import threading
import time
from typing import Union, Optional
from enum import Enum
import ray.cloudpickle as pickle
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import (
class _ExternalEnvWrapper(external_cls):

    def __init__(self, real_env):
        super().__init__(observation_space=real_env.observation_space, action_space=real_env.action_space)

    def run(self):
        time.sleep(999999)