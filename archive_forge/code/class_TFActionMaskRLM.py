import gymnasium as gym
from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.torch_utils import FLOAT_MIN
class TFActionMaskRLM(ActionMaskRLMBase, PPOTfRLModule):

    def _forward_inference(self, batch, **kwargs):
        return mask_forward_fn_tf(super()._forward_inference, batch, **kwargs)

    def _forward_train(self, batch, *args, **kwargs):
        return mask_forward_fn_tf(super()._forward_train, batch, **kwargs)

    def _forward_exploration(self, batch, *args, **kwargs):
        return mask_forward_fn_tf(super()._forward_exploration, batch, **kwargs)