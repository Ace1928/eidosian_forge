import gymnasium as gym
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
def _check_batch(batch):
    """Check whether the batch contains the required keys."""
    if 'action_mask' not in batch[SampleBatch.OBS]:
        raise ValueError("Action mask not found in observation. This model requires the environment to provide observations that include an action mask (i.e. an observation space of the Dict space type that looks as follows: \n{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),'observations': <observation_space>}")
    if 'observations' not in batch[SampleBatch.OBS]:
        raise ValueError("Observations not found in observation.This model requires the environment to provide observations that include a  (i.e. an observation space of the Dict space type that looks as follows: \n{'action_mask': Box(0.0, 1.0, shape=(self.action_space.n,)),'observations': <observation_space>}")