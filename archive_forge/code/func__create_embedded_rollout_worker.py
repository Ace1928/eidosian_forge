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
def _create_embedded_rollout_worker(kwargs, send_fn):
    """Create a local rollout worker and a thread that samples from it.

    Args:
        kwargs: Args for the RolloutWorker constructor.
        send_fn: Function to send a JSON request to the server.
    """
    kwargs = kwargs.copy()
    kwargs['config'] = kwargs['config'].copy(copy_frozen=False)
    config = kwargs['config']
    config.output = None
    config.input_ = 'sampler'
    config.input_config = {}
    if config.env is None:
        from ray.rllib.examples.env.random_env import RandomEnv, RandomMultiAgentEnv
        env_config = {'action_space': config.action_space, 'observation_space': config.observation_space}
        is_ma = config.is_multi_agent()
        kwargs['env_creator'] = _auto_wrap_external(lambda _: (RandomMultiAgentEnv if is_ma else RandomEnv)(env_config))
    else:
        real_env_creator = kwargs['env_creator']
        kwargs['env_creator'] = _auto_wrap_external(real_env_creator)
    logger.info('Creating rollout worker with kwargs={}'.format(kwargs))
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    rollout_worker = RolloutWorker(**kwargs)
    inference_thread = _LocalInferenceThread(rollout_worker, send_fn)
    inference_thread.start()
    return (rollout_worker, inference_thread)