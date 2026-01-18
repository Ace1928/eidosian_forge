import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
import ray.experimental.tf_utils
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy, PolicyState, PolicySpec
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _build_signature_def(self):
    """Build signature def map for tensorflow SavedModelBuilder."""
    input_signature = self._extra_input_signature_def()
    input_signature['observations'] = tf1.saved_model.utils.build_tensor_info(self._obs_input)
    if self._seq_lens is not None:
        input_signature[SampleBatch.SEQ_LENS] = tf1.saved_model.utils.build_tensor_info(self._seq_lens)
    if self._prev_action_input is not None:
        input_signature['prev_action'] = tf1.saved_model.utils.build_tensor_info(self._prev_action_input)
    if self._prev_reward_input is not None:
        input_signature['prev_reward'] = tf1.saved_model.utils.build_tensor_info(self._prev_reward_input)
    input_signature['is_training'] = tf1.saved_model.utils.build_tensor_info(self._is_training)
    if self._timestep is not None:
        input_signature['timestep'] = tf1.saved_model.utils.build_tensor_info(self._timestep)
    for state_input in self._state_inputs:
        input_signature[state_input.name] = tf1.saved_model.utils.build_tensor_info(state_input)
    output_signature = self._extra_output_signature_def()
    for i, a in enumerate(tf.nest.flatten(self._sampled_action)):
        output_signature['actions_{}'.format(i)] = tf1.saved_model.utils.build_tensor_info(a)
    for state_output in self._state_outputs:
        output_signature[state_output.name] = tf1.saved_model.utils.build_tensor_info(state_output)
    signature_def = tf1.saved_model.signature_def_utils.build_signature_def(input_signature, output_signature, tf1.saved_model.signature_constants.PREDICT_METHOD_NAME)
    signature_def_key = tf1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    signature_def_map = {signature_def_key: signature_def}
    return signature_def_map