import gymnasium as gym
from typing import Optional, List, Dict
from ray.rllib.algorithms.sac.sac_torch_model import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@override(SACTorchModel)
def _get_q_value(self, model_out: TensorType, actions, net, state_in: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
    if actions is not None and (not model_out.get('obs_and_action_concatenated') is True):
        model_out['obs_and_action_concatenated'] = True
        if self.concat_obs_and_actions:
            model_out[SampleBatch.OBS] = torch.cat([model_out[SampleBatch.OBS], actions], dim=-1)
        else:
            model_out[SampleBatch.OBS] = force_list(model_out[SampleBatch.OBS]) + [actions]
    model_out['is_training'] = True
    out, state_out = net(model_out, state_in, seq_lens)
    return (out, state_out)