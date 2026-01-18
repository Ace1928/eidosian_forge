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
def _setup_local_rollout_worker(self, update_interval):
    self.update_interval = update_interval
    self.last_updated = 0
    logger.info('Querying server for rollout worker settings.')
    kwargs = self._send({'command': Commands.GET_WORKER_ARGS})['worker_args']
    self.rollout_worker, self.inference_thread = _create_embedded_rollout_worker(kwargs, self._send)
    self.env = self.rollout_worker.env