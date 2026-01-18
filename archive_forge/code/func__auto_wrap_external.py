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
def _auto_wrap_external(real_env_creator):
    """Wrap an environment in the ExternalEnv interface if needed.

    Args:
        real_env_creator: Create an env given the env_config.
    """

    def wrapped_creator(env_config):
        real_env = real_env_creator(env_config)
        if not isinstance(real_env, (ExternalEnv, ExternalMultiAgentEnv)):
            logger.info('The env you specified is not a supported (sub-)type of ExternalEnv. Attempting to convert it automatically to ExternalEnv.')
            if isinstance(real_env, MultiAgentEnv):
                external_cls = ExternalMultiAgentEnv
            else:
                external_cls = ExternalEnv

            class _ExternalEnvWrapper(external_cls):

                def __init__(self, real_env):
                    super().__init__(observation_space=real_env.observation_space, action_space=real_env.action_space)

                def run(self):
                    time.sleep(999999)
            return _ExternalEnvWrapper(real_env)
        return real_env
    return wrapped_creator