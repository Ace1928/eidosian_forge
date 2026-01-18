import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _initialize_loss_from_dummy_batch(self, auto_remove_unneeded_view_reqs: bool=True, stats_fn=None) -> None:
    """Performs test calls through policy's model and loss.

        NOTE: This base method should work for define-by-run Policies such as
        torch and tf-eager policies.

        If required, will thereby detect automatically, which data views are
        required by a) the forward pass, b) the postprocessing, and c) the loss
        functions, and remove those from self.view_requirements that are not
        necessary for these computations (to save data storage and transfer).

        Args:
            auto_remove_unneeded_view_reqs: Whether to automatically
                remove those ViewRequirements records from
                self.view_requirements that are not needed.
            stats_fn (Optional[Callable[[Policy, SampleBatch], Dict[str,
                TensorType]]]): An optional stats function to be called after
                the loss.
        """
    if self.config.get('_disable_initialize_loss_from_dummy_batch', False):
        return
    self._no_tracing = True
    global_ts_before_init = int(convert_to_numpy(self.global_timestep))
    sample_batch_size = min(max(self.batch_divisibility_req * 4, 32), self.config['train_batch_size'])
    self._dummy_batch = self._get_dummy_batch_from_view_requirements(sample_batch_size)
    self._lazy_tensor_dict(self._dummy_batch)
    explore = self.config.get('_enable_new_api_stack', False)
    actions, state_outs, extra_outs = self.compute_actions_from_input_dict(self._dummy_batch, explore=explore)
    if not self.config.get('_enable_new_api_stack', False):
        for key, view_req in self.view_requirements.items():
            if key not in self._dummy_batch.accessed_keys:
                view_req.used_for_compute_actions = False
    for key, value in extra_outs.items():
        self._dummy_batch[key] = value
        if key not in self.view_requirements:
            if isinstance(value, (dict, np.ndarray)):
                space = get_gym_space_from_struct_of_tensors(value)
                self.view_requirements[key] = ViewRequirement(space=space, used_for_compute_actions=False)
            else:
                raise ValueError('policy.compute_actions_from_input_dict() returns an extra action output that is neither a numpy array nor a dict.')
    for key in self._dummy_batch.accessed_keys:
        if key not in self.view_requirements:
            self.view_requirements[key] = ViewRequirement()
            self.view_requirements[key].used_for_compute_actions = False
    new_batch = self._get_dummy_batch_from_view_requirements(sample_batch_size)
    self._dummy_batch.set_get_interceptor(None)
    for k in new_batch:
        if k not in self._dummy_batch:
            self._dummy_batch[k] = new_batch[k]
    self._dummy_batch.accessed_keys.clear()
    self._dummy_batch.deleted_keys.clear()
    self._dummy_batch.added_keys.clear()
    if self.exploration:
        self.exploration.postprocess_trajectory(self, self._dummy_batch)
    postprocessed_batch = self.postprocess_trajectory(self._dummy_batch)
    seq_lens = None
    if state_outs:
        B = 4
        if self.config.get('_enable_new_api_stack', False):
            sub_batch = postprocessed_batch[:B]
            postprocessed_batch['state_in'] = sub_batch['state_in']
            postprocessed_batch['state_out'] = sub_batch['state_out']
        else:
            i = 0
            while 'state_in_{}'.format(i) in postprocessed_batch:
                postprocessed_batch['state_in_{}'.format(i)] = postprocessed_batch['state_in_{}'.format(i)][:B]
                if 'state_out_{}'.format(i) in postprocessed_batch:
                    postprocessed_batch['state_out_{}'.format(i)] = postprocessed_batch['state_out_{}'.format(i)][:B]
                i += 1
        seq_len = sample_batch_size // B
        seq_lens = np.array([seq_len for _ in range(B)], dtype=np.int32)
        postprocessed_batch[SampleBatch.SEQ_LENS] = seq_lens
    if not self.config.get('_enable_new_api_stack'):
        train_batch = self._lazy_tensor_dict(postprocessed_batch)
        train_batch.set_training(True)
        if seq_lens is not None:
            train_batch[SampleBatch.SEQ_LENS] = seq_lens
        train_batch.count = self._dummy_batch.count
        if self._loss is not None:
            self._loss(self, self.model, self.dist_class, train_batch)
        elif is_overridden(self.loss) and (not self.config['in_evaluation']):
            self.loss(self.model, self.dist_class, train_batch)
        if stats_fn is not None:
            stats_fn(self, train_batch)
        if hasattr(self, 'stats_fn') and (not self.config['in_evaluation']):
            self.stats_fn(train_batch)
    else:
        for key in set(postprocessed_batch.keys()).difference(set(new_batch.keys())):
            if key not in self.view_requirements and key != SampleBatch.SEQ_LENS:
                self.view_requirements[key] = ViewRequirement(used_for_compute_actions=False)
    self._no_tracing = False
    if not self.config.get('_enable_new_api_stack') and auto_remove_unneeded_view_reqs:
        all_accessed_keys = train_batch.accessed_keys | self._dummy_batch.accessed_keys | self._dummy_batch.added_keys
        for key in all_accessed_keys:
            if key not in self.view_requirements and key != SampleBatch.SEQ_LENS:
                self.view_requirements[key] = ViewRequirement(used_for_compute_actions=False)
        if self._loss or is_overridden(self.loss):
            for key in self._dummy_batch.accessed_keys:
                if key not in train_batch.accessed_keys and key in self.view_requirements and (key not in self.model.view_requirements) and (key not in [SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.UNROLL_ID, SampleBatch.TERMINATEDS, SampleBatch.TRUNCATEDS, SampleBatch.REWARDS, SampleBatch.INFOS, SampleBatch.T]):
                    self.view_requirements[key].used_for_training = False
            for key in list(self.view_requirements.keys()):
                if key not in all_accessed_keys and key not in [SampleBatch.EPS_ID, SampleBatch.AGENT_INDEX, SampleBatch.UNROLL_ID, SampleBatch.TERMINATEDS, SampleBatch.TRUNCATEDS, SampleBatch.REWARDS, SampleBatch.INFOS, SampleBatch.T] and (key not in self.model.view_requirements):
                    if key in self._dummy_batch.deleted_keys:
                        logger.warning("SampleBatch key '{}' was deleted manually in postprocessing function! RLlib will automatically remove non-used items from the data stream. Remove the `del` from your postprocessing function.".format(key))
                    elif self.config['output'] is None:
                        del self.view_requirements[key]
    if type(self.global_timestep) is int:
        self.global_timestep = global_ts_before_init
    elif isinstance(self.global_timestep, tf.Variable):
        self.global_timestep.assign(global_ts_before_init)
    else:
        raise ValueError('Variable self.global_timestep of policy {} needs to be either of type `int` or `tf.Variable`, but is of type {}.'.format(self, type(self.global_timestep)))