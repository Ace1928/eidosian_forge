import gymnasium as gym
import logging
import numpy as np
from typing import Dict, List, Optional, Type, Union
import ray
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import (
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import TensorType
class ImpalaTorchPolicy(VTraceOptimizer, LearningRateSchedule, EntropyCoeffSchedule, ValueNetworkMixin, TorchPolicyV2):
    """PyTorch policy class used with Impala."""

    def __init__(self, observation_space, action_space, config):
        config = dict(ray.rllib.algorithms.impala.impala.ImpalaConfig().to_dict(), **config)
        if not config.get('_enable_new_api_stack'):
            VTraceOptimizer.__init__(self)
            LearningRateSchedule.__init__(self, config['lr'], config['lr_schedule'])
            EntropyCoeffSchedule.__init__(self, config['entropy_coeff'], config['entropy_coeff_schedule'])
        TorchPolicyV2.__init__(self, observation_space, action_space, config, max_seq_len=config['model']['max_seq_len'])
        ValueNetworkMixin.__init__(self, config)
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def loss(self, model: ModelV2, dist_class: Type[ActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)
        if isinstance(self.action_space, gym.spaces.Discrete):
            is_multidiscrete = False
            output_hidden_shape = [self.action_space.n]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            is_multidiscrete = True
            output_hidden_shape = self.action_space.nvec.astype(np.int32)
        else:
            is_multidiscrete = False
            output_hidden_shape = 1

        def _make_time_major(*args, **kw):
            return make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), *args, **kw)
        actions = train_batch[SampleBatch.ACTIONS]
        dones = train_batch[SampleBatch.TERMINATEDS]
        rewards = train_batch[SampleBatch.REWARDS]
        behaviour_action_logp = train_batch[SampleBatch.ACTION_LOGP]
        behaviour_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
        if isinstance(output_hidden_shape, (list, tuple, np.ndarray)):
            unpacked_behaviour_logits = torch.split(behaviour_logits, list(output_hidden_shape), dim=1)
            unpacked_outputs = torch.split(model_out, list(output_hidden_shape), dim=1)
        else:
            unpacked_behaviour_logits = torch.chunk(behaviour_logits, output_hidden_shape, dim=1)
            unpacked_outputs = torch.chunk(model_out, output_hidden_shape, dim=1)
        values = model.value_function()
        values_time_major = _make_time_major(values)
        bootstrap_values_time_major = _make_time_major(train_batch[SampleBatch.VALUES_BOOTSTRAPPED])
        bootstrap_value = bootstrap_values_time_major[-1]
        if self.is_recurrent():
            max_seq_len = torch.max(train_batch[SampleBatch.SEQ_LENS])
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            mask = torch.reshape(mask_orig, [-1])
        else:
            mask = torch.ones_like(rewards)
        loss_actions = actions if is_multidiscrete else torch.unsqueeze(actions, dim=1)
        loss = VTraceLoss(actions=_make_time_major(loss_actions), actions_logp=_make_time_major(action_dist.logp(actions)), actions_entropy=_make_time_major(action_dist.entropy()), dones=_make_time_major(dones), behaviour_action_logp=_make_time_major(behaviour_action_logp), behaviour_logits=_make_time_major(unpacked_behaviour_logits), target_logits=_make_time_major(unpacked_outputs), discount=self.config['gamma'], rewards=_make_time_major(rewards), values=values_time_major, bootstrap_value=bootstrap_value, dist_class=TorchCategorical if is_multidiscrete else dist_class, model=model, valid_mask=_make_time_major(mask), config=self.config, vf_loss_coeff=self.config['vf_loss_coeff'], entropy_coeff=self.entropy_coeff, clip_rho_threshold=self.config['vtrace_clip_rho_threshold'], clip_pg_rho_threshold=self.config['vtrace_clip_pg_rho_threshold'])
        model.tower_stats['pi_loss'] = loss.pi_loss
        model.tower_stats['vf_loss'] = loss.vf_loss
        model.tower_stats['entropy'] = loss.entropy
        model.tower_stats['mean_entropy'] = loss.mean_entropy
        model.tower_stats['total_loss'] = loss.total_loss
        values_batched = make_time_major(self, train_batch.get(SampleBatch.SEQ_LENS), values)
        model.tower_stats['vf_explained_var'] = explained_variance(torch.reshape(loss.value_targets, [-1]), torch.reshape(values_batched, [-1]))
        if self.config.get('_separate_vf_optimizer'):
            return (loss.loss_wo_vf, loss.vf_loss)
        else:
            return loss.total_loss

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy({'cur_lr': self.cur_lr, 'total_loss': torch.mean(torch.stack(self.get_tower_stats('total_loss'))), 'policy_loss': torch.mean(torch.stack(self.get_tower_stats('pi_loss'))), 'entropy': torch.mean(torch.stack(self.get_tower_stats('mean_entropy'))), 'entropy_coeff': self.entropy_coeff, 'var_gnorm': global_norm(self.model.trainable_variables()), 'vf_loss': torch.mean(torch.stack(self.get_tower_stats('vf_loss'))), 'vf_explained_var': torch.mean(torch.stack(self.get_tower_stats('vf_explained_var')))})

    @override(TorchPolicyV2)
    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[SampleBatch]=None, episode: Optional['Episode']=None):
        if self.config['vtrace']:
            sample_batch = compute_bootstrap_value(sample_batch, self)
        return sample_batch

    @override(TorchPolicyV2)
    def extra_grad_process(self, optimizer: 'torch.optim.Optimizer', loss: TensorType) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def get_batch_divisibility_req(self) -> int:
        return self.config['rollout_fragment_length']