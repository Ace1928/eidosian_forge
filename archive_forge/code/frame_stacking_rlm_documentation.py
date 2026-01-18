from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
import gymnasium as gym
An RLModules that takes the last n observations as input.

    The idea behind this model is to demonstrate how we can modify an existing RLModule
    with a custom view requirement. In this case, we hack a PPORModule so that it
    constructs its models for an observation space that is num_frames times larger than
    the original observation space. We then stack the last num_frames observations on
    top of each other and feed them into the encoder. This allows us to train a model
    that can make use of the temporal information in the observations.
    