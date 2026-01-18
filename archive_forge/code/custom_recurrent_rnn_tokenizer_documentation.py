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
Example of define custom tokenizers for recurrent models in RLModules.

This example shows the following steps:
- Define a custom tokenizer for a recurrent encoder.
- Define a model config that builds the custom tokenizer.
- Modify the default PPOCatalog to use the custom tokenizer config.
- Run a training that uses the custom tokenizer.
