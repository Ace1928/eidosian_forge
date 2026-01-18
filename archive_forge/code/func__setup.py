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
def _setup(self, *, validate_env: Optional[Callable[[EnvType], None]]=None, config: Optional['AlgorithmConfig']=None, num_workers: int=0, local_worker: bool=True):
    """Initializes a WorkerSet instance.
        Args:
            validate_env: Optional callable to validate the generated
                environment (only on worker=0).
            config: Optional dict that extends the common config of
                the Algorithm class.
            num_workers: Number of remote rollout workers to create.
            local_worker: Whether to create a local (non @ray.remote) worker
                in the returned set as well (default: True). If `num_workers`
                is 0, always create a local worker.
        """
    self._local_worker = None
    if num_workers == 0:
        local_worker = True
    local_tf_session_args = config.tf_session_args.copy()
    local_tf_session_args.update(config.local_tf_session_args)
    self._local_config = config.copy(copy_frozen=False).framework(tf_session_args=local_tf_session_args)
    if config.input_ == 'dataset':
        self._ds, self._ds_shards = get_dataset_and_shards(config, num_workers)
    else:
        self._ds = None
        self._ds_shards = None
    self.add_workers(num_workers, validate=config.validate_workers_after_construction)
    if local_worker and self.__worker_manager.num_actors() > 0 and (not config.create_env_on_local_worker) and (not config.observation_space or not config.action_space):
        spaces = self._get_spaces_from_remote_worker()
    else:
        spaces = None
    if local_worker:
        self._local_worker = self._make_worker(cls=self.env_runner_cls, env_creator=self._env_creator, validate_env=validate_env, worker_index=0, num_workers=num_workers, config=self._local_config, spaces=spaces)