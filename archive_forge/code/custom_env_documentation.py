import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random
import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config.