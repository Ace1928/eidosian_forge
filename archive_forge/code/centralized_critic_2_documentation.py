import numpy as np
from gymnasium.spaces import Dict, Discrete
import argparse
import os
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.models.centralized_critic_models import (
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
Rewrites the agent obs to include opponent data for training.