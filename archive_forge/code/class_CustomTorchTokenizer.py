import argparse
import os
import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.sample_batch import SampleBatch
from dataclasses import dataclass
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.tf.base import TfModel
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.core.models.configs import ModelConfig
class CustomTorchTokenizer(TorchModel, Encoder):

    def __init__(self, config) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.net = nn.Sequential(nn.Linear(config.input_dims[0], config.output_dims[0]))

    def get_output_specs(self):
        output_dim = self.config.output_dims[0]
        return SpecDict({ENCODER_OUT: TensorSpec('b, d', d=output_dim, framework='torch')})

    def _forward(self, inputs: dict, **kwargs):
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}