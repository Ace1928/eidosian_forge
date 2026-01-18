import logging
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from typing import (
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv, convert_to_base_env
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.offline import InputReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy, make_action_immutable
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _env_runner(worker: 'RolloutWorker', base_env: BaseEnv, extra_batch_callback: Callable[[SampleBatchType], None], normalize_actions: bool, clip_actions: bool, multiple_episodes_in_batch: bool, callbacks: 'DefaultCallbacks', perf_stats: _PerfStats, observation_fn: 'ObservationFunction', sample_collector: Optional[SampleCollector]=None, render: bool=None) -> Iterator[SampleBatchType]:
    """This implements the common experience collection logic.

    Args:
        worker: Reference to the current rollout worker.
        base_env: Env implementing BaseEnv.
        extra_batch_callback: function to send extra batch data to.
        multiple_episodes_in_batch: Whether to pack multiple
            episodes into each batch. This guarantees batches will be exactly
            `rollout_fragment_length` in size.
        normalize_actions: Whether to normalize actions to the action
            space's bounds.
        clip_actions: Whether to clip actions to the space range.
        callbacks: User callbacks to run on episode events.
        perf_stats: Record perf stats into this object.
        observation_fn: Optional multi-agent
            observation func to use for preprocessing observations.
        sample_collector: An optional
            SampleCollector object to use.
        render: Whether to try to render the environment after each
            step.

    Yields:
        Object containing state, action, reward, terminal condition,
        and other fields as dictated by `policy`.
    """
    simple_image_viewer: Optional['SimpleImageViewer'] = None

    def _new_episode(env_id):
        episode = Episode(worker.policy_map, worker.policy_mapping_fn, lambda: None, extra_batch_callback, env_id=env_id, worker=worker)
        return episode
    active_episodes: Dict[EnvID, Episode] = _NewEpisodeDefaultDict(_new_episode)
    for env_id, sub_env in base_env.get_sub_environments(as_dict=True).items():
        _create_episode(active_episodes, env_id, callbacks, worker, base_env)
    while True:
        perf_stats.incr('iters', 1)
        t0 = time.time()
        unfiltered_obs, rewards, terminateds, truncateds, infos, off_policy_actions = base_env.poll()
        env_poll_time = time.time() - t0
        if log_once('env_returns'):
            logger.info('Raw obs from env: {}'.format(summarize(unfiltered_obs)))
            logger.info('Info return from env: {}'.format(summarize(infos)))
        t1 = time.time()
        active_envs, to_eval, outputs = _process_observations(worker=worker, base_env=base_env, active_episodes=active_episodes, unfiltered_obs=unfiltered_obs, rewards=rewards, terminateds=terminateds, truncateds=truncateds, infos=infos, multiple_episodes_in_batch=multiple_episodes_in_batch, callbacks=callbacks, observation_fn=observation_fn, sample_collector=sample_collector)
        perf_stats.incr('raw_obs_processing_time', time.time() - t1)
        for o in outputs:
            yield o
        t2 = time.time()
        eval_results = _do_policy_eval(to_eval=to_eval, policies=worker.policy_map, sample_collector=sample_collector, active_episodes=active_episodes)
        perf_stats.incr('inference_time', time.time() - t2)
        t3 = time.time()
        actions_to_send: Dict[EnvID, Dict[AgentID, EnvActionType]] = _process_policy_eval_results(to_eval=to_eval, eval_results=eval_results, active_episodes=active_episodes, active_envs=active_envs, off_policy_actions=off_policy_actions, policies=worker.policy_map, normalize_actions=normalize_actions, clip_actions=clip_actions)
        perf_stats.incr('action_processing_time', time.time() - t3)
        t4 = time.time()
        base_env.send_actions(actions_to_send)
        perf_stats.incr('env_wait_time', env_poll_time + time.time() - t4)
        if render:
            t5 = time.time()
            rendered = base_env.try_render()
            if isinstance(rendered, np.ndarray) and len(rendered.shape) == 3:
                if simple_image_viewer is None:
                    try:
                        from gymnasium.envs.classic_control.rendering import SimpleImageViewer
                        simple_image_viewer = SimpleImageViewer()
                    except (ImportError, ModuleNotFoundError):
                        render = False
                        logger.warning('Could not import gymnasium.envs.classic_control.rendering! Try `pip install gymnasium[all]`.')
                if simple_image_viewer:
                    simple_image_viewer.imshow(rendered)
            elif rendered not in [True, False, None]:
                raise ValueError(f"The env's ({base_env}) `try_render()` method returned an unsupported value! Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.")
            perf_stats.incr('env_render_time', time.time() - t5)