import gymnasium as gym
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
class ActionMaskRLMBase(RLModule):

    def __init__(self, config: RLModuleConfig):
        if not isinstance(config.observation_space, gym.spaces.Dict):
            raise ValueError('This model requires the environment to provide a gym.spaces.Dict observation space.')
        config.observation_space = config.observation_space['observations']
        super().__init__(config)