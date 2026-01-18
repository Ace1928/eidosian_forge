import functools
import gymnasium as gym
import logging
import importlib.util
import os
from typing import (
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import (
def _get_spaces_from_remote_worker(self):
    """Infer observation and action spaces from a remote worker.

        Returns:
            A dict mapping from policy ids to spaces.
        """
    worker_id = self.__worker_manager.actor_ids()[0]
    if issubclass(self.env_runner_cls, RolloutWorker):
        remote_spaces = self.foreach_worker(lambda worker: worker.foreach_policy(lambda p, pid: (pid, p.observation_space, p.action_space)), remote_worker_ids=[worker_id], local_worker=False)
    else:
        remote_spaces = self.foreach_worker(lambda worker: worker.marl_module.foreach_module(lambda mid, m: (mid, m.config.observation_space, m.config.action_space)) if hasattr(worker, 'marl_module') else [(DEFAULT_POLICY_ID, worker.module.config.observation_space, worker.module.config.action_space)])
    if not remote_spaces:
        raise ValueError('Could not get observation and action spaces from remote worker. Maybe specify them manually in the config?')
    spaces = {e[0]: (getattr(e[1], 'original_space', e[1]), e[2]) for e in remote_spaces[0]}
    if issubclass(self.env_runner_cls, RolloutWorker):
        env_spaces = self.foreach_worker(lambda worker: worker.foreach_env(lambda env: (env.observation_space, env.action_space)), remote_worker_ids=[worker_id], local_worker=False)
        if env_spaces:
            spaces['__env__'] = env_spaces[0][0]
    logger.info(f'Inferred observation/action spaces from remote worker (local worker has no env): {spaces}')
    return spaces