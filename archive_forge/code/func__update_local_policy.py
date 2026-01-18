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
def _update_local_policy(self, force=False):
    assert self.inference_thread.is_alive()
    if self.update_interval and time.time() - self.last_updated > self.update_interval or force:
        logger.info('Querying server for new policy weights.')
        resp = self._send({'command': Commands.GET_WEIGHTS})
        weights = resp['weights']
        global_vars = resp['global_vars']
        logger.info('Updating rollout worker weights and global vars {}.'.format(global_vars))
        self.rollout_worker.set_weights(weights, global_vars)
        self.last_updated = time.time()