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
def _do_policy_eval(*, to_eval: Dict[PolicyID, List[_PolicyEvalData]], policies: PolicyMap, sample_collector: SampleCollector, active_episodes: Dict[EnvID, Episode]) -> Dict[PolicyID, Tuple[TensorStructType, StateBatch, dict]]:
    """Call compute_actions on collected episode/model data to get next action.

    Args:
        to_eval: Mapping of policy IDs to lists of _PolicyEvalData objects
            (items in these lists will be the batch's items for the model
            forward pass).
        policies: Mapping from policy ID to Policy obj.
        sample_collector: The SampleCollector object to use.
        active_episodes: Mapping of EnvID to its currently active episode.

    Returns:
        Dict mapping PolicyIDs to compute_actions_from_input_dict() outputs.
    """
    eval_results: Dict[PolicyID, TensorStructType] = {}
    if log_once('compute_actions_input'):
        logger.info('Inputs to compute_actions():\n\n{}\n'.format(summarize(to_eval)))
    for policy_id, eval_data in to_eval.items():
        try:
            policy: Policy = _get_or_raise(policies, policy_id)
        except ValueError:
            episode = active_episodes[eval_data[0].env_id]
            _assert_episode_not_faulty(episode)
            policy_id = episode.policy_mapping_fn(eval_data[0].agent_id, episode, worker=episode.worker)
            policy: Policy = _get_or_raise(policies, policy_id)
        input_dict = sample_collector.get_inference_input_dict(policy_id)
        eval_results[policy_id] = policy.compute_actions_from_input_dict(input_dict, timestep=policy.global_timestep, episodes=[active_episodes[t.env_id] for t in eval_data])
    if log_once('compute_actions_result'):
        logger.info('Outputs of compute_actions():\n\n{}\n'.format(summarize(eval_results)))
    return eval_results